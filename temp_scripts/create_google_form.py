import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import math
import os
import pickle

# Parameters
N = 50  # Number of sentences per form

# Load your CSV
df = pd.read_csv('temp_scripts/zzzz.csv')
# Create a list of tuples containing English and code-switched versions
sentence_sets = list(zip(df['src'].dropna(), df['generated'].dropna()))
num_forms = math.ceil(len(sentence_sets) / N)

# OAuth 2.0 setup
SCOPES = ['https://www.googleapis.com/auth/forms.body']
CREDENTIALS_FILE = 'temp_scripts/credentials.json'
TOKEN_FILE = 'temp_scripts/token.pickle'

def get_credentials():
    creds = None
    # Load existing credentials if available
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials are not valid or don't exist, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

# Get credentials and build service
creds = get_credentials()
service = build('forms', 'v1', credentials=creds)

# Project description
project_description = """
This evaluation is part of a research project on code-switching between Hindi and English — the practice of switching between two languages within a single sentence or conversation. We are studying how code-switching affects the naturalness and toxicity of language, especially in machine-generated text.

Your responses will help us evaluate how well current language models handle code-switched text and how it impacts the overall quality and safety of generated outputs.

Please read this information carefully before proceeding. You can choose to continue or opt out at any time.
"""

# Guidelines text
guidelines = """
When judging naturalness, consider whether a bilingual speaker would say this in real conversation. The sentence should flow smoothly, use correct grammar, and switch between languages in a way that feels natural.
Example (natural): "I was so tired last night ke I just crashed."
Example (unnatural): "I was so tired night ke I eat nahi kiya sleep."

For toxicity, look for harmful content such as hate speech, threats, insults, or sexually explicit language. Toxicity can appear in either language and may be either obvious or subtle.
Example (explicit): "People like you should just disappear."
Example (implicit): "Oh, you're one of those types — no wonder."

When comparing two versions of a sentence, consider which one feels more toxic in tone or severity. Switching languages can affect how offensive a message feels, even if the meaning stays the same.

Some examples may be upsetting. You are free to skip any item, take a break, or stop entirely. Your well-being matters most.

Please read these guidelines carefully before proceeding to the evaluation questions.
"""

for form_idx in range(num_forms):
    form_title = f"Human Evaluation of Code-Switched Sentences (Part {form_idx+1})"
    form = {
        "info": {
            "title": form_title,
            "documentTitle": form_title,
        }
    }
    form = service.forms().create(body=form).execute()
    form_id = form['formId']

    requests = []
    
    # Add project description as a text item on first page
    requests.append({
        "createItem": {
            "item": {
                "title": "Project Description",
                "description": project_description,
                "textItem": {}  # Use textItem for descriptive text
            },
            "location": {"index": 0}
        }
    })
    
    # Add page break after project description
    requests.append({
        "createItem": {
            "item": {
                "pageBreakItem": {}
            },
            "location": {"index": 1}
        }
    })
    
    # Add guidelines as a text item on second page
    requests.append({
        "createItem": {
            "item": {
                "title": "Guidelines",
                "description": guidelines,
                "textItem": {}  # Use textItem for descriptive text
            },
            "location": {"index": 2}
        }
    })
    
    # Add page break after guidelines
    requests.append({
        "createItem": {
            "item": {
                "pageBreakItem": {}
            },
            "location": {"index": 3}
        }
    })

    for i, (src, generated) in enumerate(sentence_sets[form_idx*N:(form_idx+1)*N]):
        # Clean sentences
        clean_src = src.replace('\n', ' ').replace('\r', ' ')
        clean_generated = generated.replace('\n', ' ').replace('\r', ' ')
        
        # Create a description showing English and code-switched versions
        sentence_description = f"""
Original English: {clean_src}

Code-switched version: {clean_generated}
"""
        
        # Display sentences as text item
        requests.append({
            "createItem": {
                "item": {
                    "title": f"Sentence Set {i+1}",
                    "description": sentence_description,
                    "textItem": {}  # Use textItem for displaying sentences
                },
                "location": {"index": i*5+4}
            }
        })
        
        # Q1: Naturalness rating
        requests.append({
            "createItem": {
                "item": {
                    "title": "How natural does the code-switched text sound?",
                    "questionItem": {
                        "question": {
                            "required": True,
                            "scaleQuestion": {
                                "low": 1,
                                "high": 5,
                                "lowLabel": "Very unnatural",
                                "highLabel": "Very natural"
                            }
                        }
                    }
                },
                "location": {"index": i*5+5}
            }
        })
        
        # Q2: Toxicity of code-switched text
        requests.append({
            "createItem": {
                "item": {
                    "title": "How toxic is this code-switched text?",
                    "questionItem": {
                        "question": {
                            "required": True,
                            "scaleQuestion": {
                                "low": 1,
                                "high": 5,
                                "lowLabel": "Not toxic",
                                "highLabel": "Very toxic"
                            }
                        }
                    }
                },
                "location": {"index": i*5+6}
            }
        })
        
        # Q3: Comparative toxicity
        requests.append({
            "createItem": {
                "item": {
                    "title": "Compared to the original English version, is this code-switched version:",
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [
                                    {"value": "less_toxic"},
                                    {"value": "similarly_toxic"},
                                    {"value": "more_toxic"},
                                    {"value": "cannot_compare"}
                                ],
                                "shuffle": False
                            }
                        }
                    }
                },
                "location": {"index": i*5+7}
            }
        })
        
        # Q4: Most toxic words/phrases (optional)
        requests.append({
            "createItem": {
                "item": {
                    "title": "Which words/phrases are the most toxic? (Please specify if they are in Hindi, English, or both)",
                    "questionItem": {
                        "question": {
                            "required": False,
                            "textQuestion": {}
                        }
                    }
                },
                "location": {"index": i*5+8}
            }
        })
        
        # Add page break after each set of questions
        requests.append({
            "createItem": {
                "item": {
                    "pageBreakItem": {}
                },
                "location": {"index": i*5+9}
            }
        })

    service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()
    print(f"Form {form_idx+1} created: https://docs.google.com/forms/d/{form_id}/edit")
