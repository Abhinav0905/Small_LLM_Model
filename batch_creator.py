import torch
import numpy as np
from datasets import load_from_disk
from transformers import GPT2TokenizerFast


def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class ResumeBatchCreator:
    """ Input - Output batch for resume dataset """
    def __init__(self, tokenized_dataset_path, device=None):
        
        if device is None:
            device = get_best_device()
        
        self.device = device
        self.tokenized_dataset = load_from_disk(tokenized_dataset_path)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        # Convent to continous Toekn stream 
        self.create_continous_stream()
        
        print(f"dataSet loaded: {len(self.tokenized_dataset)} resumes")
        print(f"Continous Stream: {len(self.token_stream):,} tokens")
        print(f"Device {self.device}")


    def create_continous_stream(self):
        print("\n ===========CREATING CONTINUOUS TOKEN STREAM === ")

        all_tokens = []
        resume_boundaries = []  # Track where each resume starts/ends

        for i, sample in enumerate(self.tokenized_dataset):
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']

            # only take real token(not padding)
            real_tokens = []
            for token_id, mask in zip(input_ids, attention_mask):
                if mask == 1:
                    real_tokens.append(token_id)

            # Add resume seperator token
            real_tokens.append(self.tokenizer.eos_token_id)

            # Record boundaries
            start_pos = len(all_tokens)
            all_tokens.extend(real_tokens)
            end_pos = len(all_tokens)

            resume_boundaries.append((start_pos, end_pos))

            if i % 5 == 0:
               print(f"   Processed {i+1}/{len(self.tokenized_dataset)} resumes")

        self.token_stream = np.array(all_tokens, dtype=np.int64)
        self.resume_boundaries = resume_boundaries

        print(f"Total tokens: {len(self.token_stream):,}")
        print(f"Average tokens per resume: {len(self.token_stream)/len(self.tokenized_dataset):.1f}")

    
    def get_batch(self,batch_size=4,block_size=256, split='train'):

        data = self.token_stream

        # Step 1: Generate random starting positions
        max_start = len(data) - block_size - 1
        if max_start <= 0:
            raise ValueError(f"Dataset too small! Need at least {block_size + 1} tokens")
        
        # Random starting positions
        ix = torch.randint(0, max_start, (batch_size,))
        
        # Step 2: Create input sequences (x)
        x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
        
        # Step 3: Create target sequences (y) - shifted by 1
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
        
        # Step 4: Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        return x, y
    

    def demonstrate_batching(self, batch_size=2, block_size=64):
        """Show how batching works"""
        print(f"\n=== BATCH DEMONSTRATION ===")
        print(f"Batch size: {batch_size}")
        print(f"Block size: {block_size}")
        
        # Get a batch
        x, y = self.get_batch(batch_size, block_size)
        
        print(f"\nBatch shapes:")
        print(f"   Input (x): {x.shape}")  # [batch_size, block_size]
        print(f"   Target (y): {y.shape}") # [batch_size, block_size]
        
        # Show first sequence in batch
        print(f"\nFirst sequence in batch:")
        print(f"   Input IDs: {x[0][:10].tolist()}... (first 10)")
        print(f"   Target IDs: {y[0][:10].tolist()}... (first 10)")
        
        # Decode to show actual text
        input_text = self.tokenizer.decode(x[0][:20], skip_special_tokens=True)
        target_text = self.tokenizer.decode(y[0][:20], skip_special_tokens=True)
        
        print(f"\nDecoded text:")
        print(f"   Input: '{input_text}'")
        print(f"   Target: '{target_text}'")
        
        # Show the shift
        print(f"\nShift demonstration:")
        for i in range(5):
            input_token = self.tokenizer.decode([x[0][i]], skip_special_tokens=True)
            target_token = self.tokenizer.decode([y[0][i]], skip_special_tokens=True)
            print(f"   Position {i}: '{input_token}' → '{target_token}'")

    
    def analyze_token_stream(self):
        """Analyze the continuous token stream"""
        print(f"\n=== TOKEN STREAM ANALYSIS ===")
        print(f"Total tokens: {len(self.token_stream):,}")
        print(f"Unique tokens: {len(set(self.token_stream)):,}")
        print(f"Token range: {self.token_stream.min()} to {self.token_stream.max()}")
        
        # Show distribution
        unique, counts = np.unique(self.token_stream, return_counts=True)
        top_tokens = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nMost frequent tokens:")
        for token_id, count in top_tokens:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            print(f"   '{token_text}' (ID: {token_id}): {count:,} times")
        
        # Show sample from stream
        print(f"\nSample from token stream:")
        sample_tokens = self.token_stream[1000:1020]  # 20 tokens
        sample_text = self.tokenizer.decode(sample_tokens, skip_special_tokens=True)
        print(f"   Tokens: {sample_tokens}")
        print(f"   Text: '{sample_text}'")


def main():
    print("=== RESUME BATCH CREATOR ===")
    
    # Initialize
    batch_creator = ResumeBatchCreator('tokenized_dataset/tokenized_data')
    
    # Analyze token stream
    batch_creator.analyze_token_stream()
    
    # Demonstrate batching
    batch_creator.demonstrate_batching(batch_size=2, block_size=64)
    
    # Test different batch sizes
    print(f"\n=== TESTING DIFFERENT BATCH SIZES ===")
    for batch_size in [1, 2, 4]:
        for block_size in [128, 256, 512]:
            try:
                x, y = batch_creator.get_batch(batch_size, block_size)
                print(f"Batch size {batch_size}, Block size {block_size}: {x.shape}")
            except Exception as e:
                print(f"Batch size {batch_size}, Block size {block_size}: {e}")


if __name__ == "__main__":
    main()
