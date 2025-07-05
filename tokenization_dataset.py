import argparse
import torch
print(f"MPS Available: {torch.backends.mps.is_available()}")

import os
import tiktoken
import json
import numpy as np
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from collections import Counter

class TokenizationLearner:

    def __init__(self,tokenizer_type='tiktoken'):
        self.tokenizer_type = tokenizer_type
        self.tokenizer = None
        self.vocab_size = 0
        self.setup_tokenizer()

    
    def setup_tokenizer(self):
        print(f"\n====== TOKENIZER SETUP =======")

        if self.tokenizer_type == 'gpt2':
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = self.tokenizer.vocab_size
            print(f"Using GPT-2 BPE tokenizer")

        elif self.tokenizer_type == 'tiktoken':
            self.tokenizer = tiktoken.get_encoding('gpt2')
            self.vocab_size = self.tokenizer.n_vocab
            print(f"using tiktoken BPE Encoding")

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Tokenizer type: {self.tokenizer_type}")

    
    def demonstrarte_bep(self, sample_text):
        print(f"\n ====== BPE DEMONSTRATION ====")
        print(f"Sample text: {sample_text[:100]}.....")

        if self.tokenizer_type == 'gpt2':
            tokens = self.tokenizer.tokenize(sample_text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            print(f"\n1.text -> Tokens (BPE):")
            for i, (token, token_id) in enumerate(zip(tokens[:10],token_ids[:10])):
                 print(f"   '{token}' → {token_id}")
            print(f" .... (Showing first 10 of {len(tokens)} tokens)")

            print(f"\n 2.Special Tokens..")
            print(f" PAD Token: '{self.tokenizer.pad_token}' -> {self.tokenizer.pad_token_id}")
            print(f"EOS token: '{self.tokenizer.eos_token}' -> {self.tokenizer.eos_token_id}")

            # Show Decoding
            print(f"\n3. Token -> Text ( Decoding)")
            decoded = self.tokenizer.decode(token_ids[:10])
            print(f"Showing the firts 10 decoded token {decoded}")

        elif self.tokenizer_type == 'tiktoken':
            tokens = self.tokenizer.encode(sample_text)
            print(f"\n1. Text → Token IDs:")
            print(f"   {tokens[:15]}... (showing first 15 of {len(tokens)} tokens)")
            
            decoded = self.tokenizer.decode(tokens[:15])
            print(f"\n2. Token IDs → Text:")
            print(f"   '{decoded}'")

    
    def analyze_vocabulary(self, dataset):
        """Analyze vocabulary usage in dataset"""
        print(f"\n=== VOCABULARY ANALYSIS ===")

        all_tokens = []
        all_token_ids = []

        for text in dataset['text'][:5]:
            if self.tokenizer_type == 'gpt2':
                tokens = self.tokenizer.tokenize(text)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                all_tokens.extend(tokens)
                all_token_ids.extend(token_ids)

            else:
                token_ids = self.tokenizer.encode(text)
                all_token_ids.extend(token_ids)

        # Token Frequence
        token_freq = Counter(all_token_ids)
        print(f"Total tokens analyzed: {len(all_token_ids):,}")
        print(f"Unique tokens used: {len(token_freq):,}")
        print(f"Vocabulary coverage: {len(token_freq)/self.vocab_size*100:.1f}%")
        
        # moSt Common Tokens
        print(f"\n most common tokens:")
        for token_id, freq in token_freq.most_common(10):
            if self.tokenizer_type == 'gpt2':
                token_text = self.tokenizer.decode([token_id])
            else:
                token_text = self.tokenizer.decode([token_id])
            print(f"   '{token_text}' (ID: {token_id}): {freq} times")

        return token_freq

    
    def tokenize_dataset(self, dataset, max_length=256):
        """Tokenize dataset with detailed logging"""
        print(f"\n=== TOKENIZATION PROCESS ===")
        print(f"Max sequence length: {max_length}")

        def tokenize_function(examples):
            batch_size = len(examples['text'])
            print(f"Processing batch of {batch_size} texts...")

            if self.tokenizer_type == 'gpt2':
                result = self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                return result
            
            else:  # Tiktoken
                input_ids = []
                attention_masks = []

                for text in examples['text']:
                    tokens = self.tokenizer.encode(text)

                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    truncated = True
                else:
                    truncated = False

                # Atetntion mask

                attention_mask = [1] * len(tokens)

                # Pad if needed
                if len(tokens) < max_length:
                        padding_length = max_length - len(tokens)
                        tokens.extend([0] * padding_length)
                        attention_mask.extend([0] * padding_length)
                    
                input_ids.append(tokens)
                attention_masks.append(attention_mask)
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_masks
                }
                
    # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=10)
        
        # Analyze results
        self.analyze_tokenization_results(tokenized_dataset, max_length)
        
        return tokenized_dataset

    def analyze_tokenization_results(self, tokenized_dataset, max_length):
        """Analyze tokenization results"""
        print(f"\n=== TOKENIZATION RESULTS ===")
        
        # Calculate statistics
        actual_lengths = []
        padding_counts = []
        
        for i in range(len(tokenized_dataset)):
            attention_mask = tokenized_dataset[i]['attention_mask']
            actual_length = sum(attention_mask)
            padding_count = max_length - actual_length
            
            actual_lengths.append(actual_length)
            padding_counts.append(padding_count)
        
        print(f"Total sequences: {len(tokenized_dataset)}")
        print(f"Average actual length: {np.mean(actual_lengths):.1f} tokens")
        print(f"Average padding: {np.mean(padding_counts):.1f} tokens")
        print(f"Sequences needing truncation: {sum(1 for l in actual_lengths if l == max_length)}")
        print(f"Sequences with padding: {sum(1 for p in padding_counts if p > 0)}")
        
        # Show example
        print(f"\n Example tokenization:")
        example_idx = 0
        input_ids = tokenized_dataset[example_idx]['input_ids']
        attention_mask = tokenized_dataset[example_idx]['attention_mask']
        
        print(f"   Input IDs shape: {np.array(input_ids).shape}")
        print(f"   First 10 tokens: {input_ids[:10]}")
        print(f"   Attention mask: {attention_mask[:10]}... (first 10)")
        
        # Decode example
        if self.tokenizer_type == 'gpt2':
            decoded = self.tokenizer.decode(input_ids[:20], skip_special_tokens=True)
        else:
            decoded = self.tokenizer.decode(input_ids[:20])
        print(f"   Decoded (first 20 tokens): '{decoded}'")

def main():
    parser = argparse.ArgumentParser(description='Educational tokenization of resume dataset')
    parser.add_argument('--dataset_dir', type=str, default='raw_dataset/dataset',
                    help='Directory containing raw dataset')
    parser.add_argument('--output_dir', type=str, default='tokenized_dataset',
                    help='Output directory for tokenized dataset')
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                    choices=['gpt2', 'tiktoken'],
                    help='Tokenizer type to use')
    parser.add_argument('--max_length', type=int, default=512,
                    help='Maximum sequence length')
    parser.add_argument('--demo_mode', action='store_true',
                    help='Show detailed demonstrations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw dataset
    print("=== LOADING RAW DATASET ===")
    try:
        dataset = load_from_disk(args.dataset_dir)
        print(f"Loaded {len(dataset)} resumes")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize tokenization learner
    learner = TokenizationLearner(args.tokenizer_type)
    
    # Demonstration mode
    if args.demo_mode:
        sample_text = dataset[0]['text']
        learner.demonstrarte_bep(sample_text)
        learner.analyze_vocabulary(dataset)
    
    # Tokenize dataset
    tokenized_dataset = learner.tokenize_dataset(dataset, args.max_length)
    
    # Save tokenized dataset
    print(f"\n=== SAVING TOKENIZED DATASET ===")
    tokenized_dataset.save_to_disk(os.path.join(args.output_dir, 'tokenized_data'))
    
    # Save metadata
    metadata = {
        'tokenizer_type': args.tokenizer_type,
        'max_length': args.max_length,
        'vocab_size': learner.vocab_size,
        'num_samples': len(tokenized_dataset)
    }
    
    with open(os.path.join(args.output_dir, 'tokenization_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Tokenized dataset saved to: {args.output_dir}")
    print(f"Ready for model training!")


# tokenization = TokenizationLearner()

if __name__ == "__main__":
    main()