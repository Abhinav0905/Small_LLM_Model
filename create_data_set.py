# import torch
# print(f"MPS Available: {torch.backends.mps.is_available()}")

import argparse
import os
import docx2txt
from docx import Document
import numpy as np
from datasets import Dataset
import pandas as pd


def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        return "\n".join(text)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_text_from_doc(file_path):
    """ Extract text based on file extension"""
    try:
        text = docx2txt.process(file_path)
        if text and text.strip():
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.doc':
        return extract_text_from_doc(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return ""

def clean_text(text):
    """Remove text cleaning """
    text = ''.join(text.split())
    # Remove empty line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def analyze_dataset_statistics(texts, filenames):
    """Analuze and display dataset staticstic"""
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"\n('='*50)")

    # Basic Counts 
    print(f"Total Resume: {len(texts)}")
    print(f"Total File processed {len(filenames)}")
    print(f"Successfully processed Resumes: {len([t for t in texts if t.strip()])}")  # ✅ Fixed!
    print(f"Resumes Not processed: {len([t for t in texts if not t.strip()])}") 

    # Text length statistics
    char_lengths = [len(text) for text in texts if text.strip()]
    word_lengths = [len(text.split()) for text in texts if text.strip()]
    line_lengths = [len(text.split('\n')) for text in texts if text.strip()]
    
    if char_lengths:
        print(f"\n CHARACTER STATISTICS:")
        print(f"   Average: {np.mean(char_lengths):.0f} characters")
        print(f"   Median: {np.median(char_lengths):.0f} characters")
        print(f"   Min: {min(char_lengths)} characters")
        print(f"   Max: {max(char_lengths)} characters")
        print(f"   Std: {np.std(char_lengths):.0f} characters")
        
        print(f"\n WORD STATISTICS:")
        print(f"   Average: {np.mean(word_lengths):.0f} words")
        print(f"   Median: {np.median(word_lengths):.0f} words")
        print(f"   Min: {min(word_lengths)} words")
        print(f"   Max: {max(word_lengths)} words")
        
        print(f"\n LINE STATISTICS:")
        print(f"   Average: {np.mean(line_lengths):.0f} lines")
        print(f"   Median: {np.median(line_lengths):.0f} lines")
        print(f"   Min: {min(line_lengths)} lines")
        print(f"   Max: {max(line_lengths)} lines")
    
    return {
        'total_resumes': len(texts),
        'char_lengths': char_lengths,
        'word_lengths': word_lengths,
        'line_lengths': line_lengths
    }


def show_sample_data(texts, filenames, num_samples=2):
    """Show sample data for inspection"""
    print(f"\n{'='*50}")
    print(f"SAMPLE DATA INSPECTION")
    print(f"\n{'='*50}")
    
    for i in range(min(num_samples, len(texts))):
        print(f"\n📄 SAMPLE {i+1}: {filenames[i]}")
        print(f"   Length: {len(texts[i])} characters, {len(texts[i].split())} words")
        print(f"   Preview: {texts[i][:200]}...")
        if len(texts[i]) > 200:
            print(f"   ... (truncated, full text is {len(texts[i])} characters)")


def create_resume_dataset(resume_dir, clean_texts=True, show_samples=True):
    """create dataset form resume files with detailed analysis """
    print(f"\n{"="*50}")
    print("Creating Resume Dataset")
    print(f"\n{'='*50}")

    texts = []
    filenames = []
    successful_files = []
    failed_files = []

    print(f"scanning directory {resume_dir}")

    # Check if the director exist
    try:
        if not os.path.exists(resume_dir):
            print("Error: Directory does not eixist")
    except:
        pass

    # Get all the .doc & .docx files

    all_files = os.listdir(resume_dir)
    word_files = [f for f in all_files if f.endswith(('.doc','.docx'))]

    print(f" Found {len(all_files)} total files")
    print(f"Found {len(word_files)} total word documents")


    # Process each file 
    for i, filename in enumerate(word_files, 1):
        file_path = os.path.join(resume_dir, filename)
        print(f"📖 Processing {i}/{len(word_files)}: {filename}")
        
        text = extract_text_from_file(file_path)
        
        if text.strip():
            if clean_texts:
                text = clean_text(text)
            
            texts.append(text)
            filenames.append(filename)
            successful_files.append(filename)
            print(f"   ✅ Success: {len(text)} characters extracted")
        else:
            failed_files.append(filename)
            print(f"   ❌ Failed: No text extracted")

    # Show processing summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   ✅ Successful: {len(successful_files)}")
    print(f"   ❌ Failed: {len(failed_files)}")

    if failed_files:
        print(f"   📋 Failed files: {', '.join(failed_files)}")

    if not texts:
        print("❌ Error: No valid resume texts found!")
        return None

    # Analyze dataset
    stats = analyze_dataset_statistics(texts, filenames)

    # Show samples
    if show_samples:
        show_sample_data(texts, filenames)

    # Create dataset dictionary
    dataset_dict = {
        'text': texts,
        'filename': filenames,
        'char_length': [len(text) for text in texts],
        'word_length': [len(text.split()) for text in texts],
        'line_length': [len(text.split('\n')) for text in texts]
    }

    # Create HuggingFace dataset
    dataset = Dataset.from_dict(dataset_dict)

    print(f"\n{'='*50}")
    print(f"DATASET CREATED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"🎉 Dataset object: {dataset}")

    return dataset, stats

def save_dataset_with_metadata(dataset, stats, output_dir):
    """Save dataset with comprehensive metadata"""
    print(f"\n{'='*50}")
    print(f"SAVING DATASET")
    print(f"{'='*50}")

    # Save HuggingFace dataset
    dataset_path = os.path.join(output_dir, 'dataset')
    dataset.save_to_disk(dataset_path)
    print(f"💾 Dataset saved to: {dataset_path}")

    # Save as CSV for easy inspection
    csv_path = os.path.join(output_dir, 'resume_dataset.csv')
    df = pd.DataFrame(dataset)
    df.to_csv(csv_path, index=False)
    print(f"📊 CSV saved to: {csv_path}")

    # Save detailed statistics
    stats_path = os.path.join(output_dir, 'dataset_statistics.csv')
    stats_df = pd.DataFrame({
        'filename': dataset['filename'],
        'char_length': dataset['char_length'],
        'word_length': dataset['word_length'],
        'line_length': dataset['line_length']
    })
    stats_df.to_csv(stats_path, index=False)
    print(f"📈 Statistics saved to: {stats_path}")

    # Save summary report
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RESUME DATASET SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total resumes: {len(dataset)}\n")
        f.write(f"Total characters: {sum(stats['char_lengths']):,}\n")
        f.write(f"Total words: {sum(stats['word_lengths']):,}\n")
        f.write(f"Average characters per resume: {np.mean(stats['char_lengths']):.0f}\n")
        f.write(f"Average words per resume: {np.mean(stats['word_lengths']):.0f}\n")
        f.write(f"Median characters per resume: {np.median(stats['char_lengths']):.0f}\n")
        f.write(f"Median words per resume: {np.median(stats['word_lengths']):.0f}\n")
        f.write(f"Shortest resume: {min(stats['char_lengths'])} characters\n")
        f.write(f"Longest resume: {max(stats['char_lengths'])} characters\n")

    print(f"📋 Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Create comprehensive resume dataset')
    parser.add_argument('--resume_dir', type=str, default='Resume_data_set', 
                    help='Directory containing resume files')
    parser.add_argument('--output_dir', type=str, default='raw_dataset',
                    help='Output directory for raw dataset')
    parser.add_argument('--clean_texts', action='store_true', default=True,
                    help='Clean extracted texts')
    parser.add_argument('--show_samples', action='store_true', default=True,
                    help='Show sample data during processing')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset
    result = create_resume_dataset(
        args.resume_dir, 
        clean_texts=args.clean_texts, 
        show_samples=args.show_samples
    )

    if result is None:
        print("❌ Failed to create dataset!")
        return

    dataset, stats = result

    # Save dataset with metadata
    save_dataset_with_metadata(dataset, stats, args.output_dir)

    print(f"\n🎉 DATASET CREATION COMPLETE!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Ready for tokenization step!")

if __name__ == "__main__":
    main() 
