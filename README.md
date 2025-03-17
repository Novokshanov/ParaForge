# Parallel Corpora Generator

A command-line tool for generating parallel corpora from JSON files containing sentences with lemmas and tags.

## Overview

This tool processes JSON files containing linguistic corpora data and uses a language model to generate parallel sentences. The original sentences and their generated counterparts are saved as parallel corpora.

## Features

- Process individual JSON files or all JSON files in a directory
- Utilize GPU acceleration with vLLM for faster processing (falls back to transformers if vLLM is not available)
- Fall back to CPU if GPU is not available
- Generate parallel sentences using the Qwen-0.5b model
- Save results as JSON files with original and generated sentence pairs

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd parallel_corpora
   ```

2. Run the installation script:
   ```
   python install.py
   ```
   
   This script will:
   - Install all required dependencies
   - Try to install vLLM for faster processing
   - Fall back to transformers if vLLM installation fails
   - Provide information about your system's compatibility

   Alternatively, you can manually install the dependencies:
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

The tool expects JSON files containing linguistic corpora data with detailed word analysis. Each JSON file should have the following structure:

```json
{
  "meta": {
    "filename": "path/to/source.txt",
    "title": "Document Title",
    "author": "",
    "year_from": "2014",
    "year_to": "2014",
    "genre": "press",
    "issue": "2014.02.13",
    "original": "original",
    "orthography": "orth_norm"
  },
  "sentences": [
    {
      "words": [
        {
          "wf": "WordForm",
          "wtype": "word",
          "ana": [
            {
              "lex": "lemma",
              "gr.pos": "POS",
              "gr.number": "sg/pl",
              "gr.case": "nom/gen/dat/acc/abl/loc",
              "trans_ru": "translation",
              "trans_ru2": "alternative translation"
            }
          ]
        }
      ],
      "text": "Complete sentence text"
    }
  ]
}
```

The parser processes this JSON structure and extracts:
1. Original sentences from the "text" field
2. Processed sentences where each word is converted to its translation with grammatical tags

Supported grammatical tags include:
- Numbers: 1, 2, 3
- Gender: masc, fem
- Number: sg (singular), pl (plural)
- Case: nom, gen, dat, acc, abl, loc
- Tense: pst, prs, fut

Words with multiple translations will be formatted as: primary[alternative1, alternative2]
Grammatical tags are added in angle brackets: translation<tag1,tag2>

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