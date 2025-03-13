import pandas as pd
import requests
from config import PERSPECTIVE_API_KEY

def analyze_comments(input_file, api_key):
    # Read the generated sentences from the CSV file
    df = pd.read_csv(input_file, header=None, names=['generated'])
    comments = df['generated'].tolist()
    
    # Define the Perspective API endpoint
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    
    # Analyze each comment
    results = []
    for comment in comments:
        data = {
            'comment': {'text': comment},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {}
            }
        }
        
        response = requests.post(url, params={'key': api_key}, json=data)
        
        if response.status_code == 200:
            result = response.json()
            analysis = {
                'comment': comment,
                'toxicity': result['attributeScores']['TOXICITY']['summaryScore']['value'],
                'severe_toxicity': result['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value'],
                'identity_attack': result['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value'],
                'insult': result['attributeScores']['INSULT']['summaryScore']['value'],
                'profanity': result['attributeScores']['PROFANITY']['summaryScore']['value'],
                'threat': result['attributeScores']['THREAT']['summaryScore']['value']
            }
            results.append(analysis)
            print(f"Analyzed comment: {comment[:50]}...")
        else:
            print(f"Error analyzing comment: {comment[:50]}...")
            print(f"Response: {response.text}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    input_file = 'data/output/generated_sentences.csv'
    results = analyze_comments(input_file, PERSPECTIVE_API_KEY)
    
    # Save the results to a CSV file
    output_file = 'data/output/perspective_analysis.csv'
    results.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}") 