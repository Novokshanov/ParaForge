# ParaForge: Parallel Subcorpora Generator

A command-line tool for generating parallel subcorpora from JSON files containing sentences with lemmas and tags. As a starting point for this project we've chosen [tsakorpus](https://tsakorpus.readthedocs.io/) corpora data scructure as a baseline.

## Overview

This tool processes JSON files containing linguistic corpora data and uses a language model to generate parallel sentences. The original sentences and their generated counterparts are saved as parallel corpora.

## Features

- Process individual JSON files or all JSON files in a directory
- Utilize GPU acceleration with vLLM for faster processing (falls back to transformers if vLLM is not available)
- Fall back to CPU if GPU is not available
- Generate parallel sentences using the fine-tuned version of Qwen-0.5b model
- Save results as JSON files with original and generated sentence pairs

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd parallel_corpora
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with vLLM installation, the tool will automatically fall back to using the transformers library.

## Usage

### Basic Usage

Process all JSON files in the data directory:

```
python -m src.cli
```

### Specify a Single File

Process a specific JSON file:

```
python -m src.cli --file example.json
```

### Custom Directories

Specify custom input and output directories:

```
python -m src.cli --data-dir custom_data --output-dir custom_output
```


## Input Format

The tool expects JSON files containing corpora data with sentences, lemmas, and tags. Please proceed to [Tsakorpus documentation page] (https://tsakorpus.readthedocs.io/en/latest/categories.html) to find out more.

Example expected JSON structure:

```json
{
"sentences": [
   {
      "words":
{
    "wf": "word1"
    "ana": [
        "lex": "corpus",
        "gr.pos": "N",
        "gr.number": "pl",
        "gr.case": "acc"
    ],
    "wf": "word2"
    "ana": [
        "lex": "corpus",
        "gr.pos": "N",
        "gr.number": "pl",
        "gr.case": "acc"
    ]
}
"text": "word1 word2"
}
}
```

## Output Format

The tool generates output files in the following format:

```json
[
  {
    "original": "This is a sample sentence.",
    "generated": "This sentence is a sample."
  },
  ...
]
```

## Troubleshooting

### vLLM Installation Issues

If you encounter issues with vLLM installation:

1. The tool will automatically fall back to using transformers
2. You can try installing vLLM manually:
   ```
   pip install ninja packaging setuptools>=49.4.0
   pip install git+https://github.com/vllm-project/vllm.git
   ```
3. For Windows users, vLLM might not be fully supported. The transformers fallback should work in all cases.



## Requirements

- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- Either vLLM 0.2.0+ or transformers 4.30.0+ (the tool will use vLLM if available, otherwise fall back to transformers)
- CUDA-compatible GPU (optional, for faster processing) 