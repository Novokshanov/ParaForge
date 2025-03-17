#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.parser import parse_json_file
from src.model import generate_parallel_sentences, initialize_model

def get_json_files(data_dir: str, filename: Optional[str] = None) -> List[Path]:
    """
    Get a list of JSON files to process.
    
    Args:
        data_dir: Directory containing JSON files
        filename: Optional specific filename to process
        
    Returns:
        List of Path objects for JSON files
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)
    
    if filename:
        file_path = data_path / filename
        if not file_path.exists():
            print(f"Error: File '{filename}' does not exist in '{data_dir}'.")
            sys.exit(1)
        if not filename.endswith('.json'):
            print(f"Error: File '{filename}' is not a JSON file.")
            sys.exit(1)
        return [file_path]
    
    # Get all JSON files in the directory
    return list(data_path.glob('*.json'))

def save_parallel_corpora(original_sentences: List[str], 
                         generated_sentences: List[str], 
                         output_file: Path) -> None:
    """
    Save original and generated sentences as parallel corpora.
    
    Args:
        original_sentences: List of original sentences
        generated_sentences: List of generated sentences
        output_file: Path to save the parallel corpora
    """
    if len(original_sentences) != len(generated_sentences):
        print("Warning: Number of original and generated sentences don't match.")
    
    parallel_data = []
    for orig, gen in zip(original_sentences, generated_sentences):
        parallel_data.append({
            "original": orig,
            "generated": gen
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parallel_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved parallel corpora to {output_file}")

def process_file(file_path: Path, model, output_dir: Path) -> None:
    """
    Process a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        model: Initialized language model
        output_dir: Directory to save output
    """
    print(f"Processing {file_path}...")
    
    try:
        # Parse the JSON file to get sentences
        sentences = parse_json_file(file_path)
        
        # Generate parallel sentences using the model
        generated_sentences = generate_parallel_sentences(model, sentences)
        
        # Create output filename
        output_file = output_dir / f"{file_path.stem}_parallel.json"
        
        # Save the parallel corpora
        save_parallel_corpora(sentences, generated_sentences, output_file)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process JSON files to create parallel corpora.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSON files (default: data)")
    parser.add_argument("--output-dir", default="data/output", help="Directory to save output files (default: data/output)")
    parser.add_argument("--file", help="Specific JSON file to process (if not specified, all JSON files will be processed)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of JSON files to process
    json_files = get_json_files(args.data_dir, args.file)
    
    if not json_files:
        print(f"No JSON files found in '{args.data_dir}'.")
        sys.exit(1)
    
    # Initialize the model
    model = initialize_model()
    
    # Process each file
    for file_path in json_files:
        process_file(file_path, model, output_dir)
    
    print("All files processed successfully.")

if __name__ == "__main__":
    main() 