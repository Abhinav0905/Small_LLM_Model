import torch
print(f"MPS Available: {torch.backends.mps.is_available()}")

import os
import argparse
import numpy as np
import docx
import docx2txt
from docx import Document
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_text_from_doc(file_path):
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.doc'):
        return extract_text_from_doc(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return ""

def create_resume_dataset(resume_dir):
    texts = []
    filenames = []

    # Get all the .doc & .docx files

    for filename in os.listdir(resume_dir):
        if filename.endswith(('.doc','docx')):
            file_path = os.path.join(resume_dir, filename)
            print(f"Processing: {file_path}")

            text = extract_text_from_file(file_path)
            if text.strip():
                texts.append(text)
                filenames.append(filename)

    # Creeate dataset
    dataset_dict = {
        'text': texts,
        'filename': filenames
    }

    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset"""
    def tokenize_function(example):
        return tokenizer(
            example['text'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description='Create dataset from resume files')
    parser.add_argument('--resume_dir', type=str, default='Resume_data_set', 
                    help='Directory containing resume files')
    parser.add_argument('--output_dir', type=str, default='processed_dataset',
                    help='Output directory for processed dataset')
    parser.add_argument('--max_length', type=int, default=512,
                    help='Maximum token length')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset from resume files
    print("Creating dataset from resume files...")
    dataset = create_resume_dataset(args.resume_dir)
    print(f"Created dataset with {len(dataset)} resumes")
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)
    
    # Save dataset
    dataset.save_to_disk(os.path.join(args.output_dir, 'raw_dataset'))
    tokenized_dataset.save_to_disk(os.path.join(args.output_dir, 'tokenized_dataset'))
    
    # Save as CSV for inspection
    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(args.output_dir, 'resume_dataset.csv'), index=False)
    
    print(f"Dataset saved to {args.output_dir}")
    print(f"Sample text length: {len(dataset[0]['text'])} characters")
    print(f"Sample tokenized length: {len(tokenized_dataset[0]['input_ids'])} tokens")


if __name__ == "__main__":
    main()
