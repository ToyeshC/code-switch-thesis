# Multi-GPU Parallelization for Fluency-Toxicity Analysis

## Overview

The fluency-toxicity correlation analysis script has been enhanced with multi-GPU parallelization to significantly speed up perplexity calculations using all 4 GPUs available on a Snellius node.

## Key Improvements

### 1. Multi-GPU Perplexity Calculator
- **`MultiGPUPerplexityCalculator`**: New class that distributes perplexity calculations across multiple GPUs
- **Batch Processing**: Processes texts in optimized batches instead of one-by-one
- **Memory Optimization**: Uses half-precision (float16) to reduce memory usage
- **Automatic Scaling**: Adjusts batch sizes based on number of available GPUs

### 2. Performance Optimizations
- **4 GPU Support**: Utilizes all 4 A100 GPUs on a Snellius node
- **Optimized Batch Sizes**:
  - 4+ GPUs: batch size 32
  - 2-3 GPUs: batch size 16
  - 1 GPU: batch size 8
- **DataLoader Optimizations**: Improved worker processes and prefetching
- **Memory Management**: Automatic cleanup and cache management

### 3. Enhanced SLURM Configuration
- **4 GPUs**: `--gpus=4` instead of `--gpus=1`
- **More Memory**: 128GB RAM instead of 64GB
- **More CPUs**: 16 cores instead of 8
- **Shorter Time**: 8 hours instead of 12 hours (expected speedup)
- **GPU Monitoring**: Built-in `nvidia-smi` monitoring

## Expected Performance Improvements

Based on the parallelization:
- **~3-4x speedup** from using 4 GPUs instead of 1
- **~2x speedup** from optimized batch processing
- **Overall: ~6-8x faster execution**

## Usage

### 1. Test Multi-GPU Setup (Recommended)
```bash
# Test if multi-GPU setup works correctly
python test_multigpu_setup.py
```

### 2. Run the Analysis
```bash
# Submit the multi-GPU job
sbatch final_job_scripts/9_analyze_fluency_toxicity_correlation.sh
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor GPU usage during execution
watch nvidia-smi
```

## Bottleneck Resolution

### Original Bottlenecks:
1. **Sequential Processing**: Texts processed one-by-one
2. **Single GPU**: Only 1 GPU utilized
3. **Memory Inefficiency**: Full precision and poor memory management
4. **Small Batches**: Inefficient GPU utilization

### Solutions Implemented:
1. **Parallel Batch Processing**: Multiple texts processed simultaneously
2. **4-GPU Distribution**: Work distributed across all available GPUs
3. **Half-Precision**: Reduced memory usage with float16
4. **Optimized Batching**: Larger, GPU-count-aware batch sizes
5. **Smart Data Loading**: Optimized DataLoader with multiple workers

## Configuration Details

### SLURM Script Changes:
```bash
# Old configuration
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# New configuration
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
```

### GPU Memory Management:
- **Device Mapping**: Automatic distribution across GPUs
- **Memory Limits**: 20GB per GPU to prevent overflow
- **Fallback Strategy**: DataParallel if device mapping fails

## Troubleshooting

### If Job Fails:
1. Check GPU availability: `nvidia-smi`
2. Test setup: `python test_multigpu_setup.py`
3. Check logs in `final_job_outputs/`

### Memory Issues:
- Script automatically handles memory cleanup
- Uses half-precision to reduce memory usage
- Implements batch size fallbacks

### Performance Monitoring:
- Built-in GPU status reporting before/after execution
- Batch processing progress indicators
- Memory usage tracking

## File Structure

```
final_python_scripts/
├── analyze_fluency_toxicity_correlation.py  # Multi-GPU version
└── ...

final_job_scripts/
├── 9_analyze_fluency_toxicity_correlation.sh  # Multi-GPU SLURM script
└── ...

test_multigpu_setup.py  # GPU setup test script
```

## Expected Output

The analysis will generate the same outputs but much faster:
- `final_outputs/experiment_d/`
  - Perplexity results
  - Statistical analyses
  - Visualizations
  - Summary report

With estimated completion time reduced from 6+ hours to 1-2 hours with 4 GPUs. 