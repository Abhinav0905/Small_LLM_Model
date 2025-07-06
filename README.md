# Small_LLM_Model
# Small Language Model (SLM) from Scratch

A GPT-style transformer implementation built from scratch in PyTorch, trained on a custom resume dataset.

## 🏆 Key Achievements
- **15.2M parameters** GPT-style transformer
- **97% accuracy** (perplexity: 1.03) on text generation
- **Professional-grade** resume text generation
- **Optimized** for Apple M2 GPU training

## 🚀 Features
- Multi-head attention mechanism
- Custom tokenization pipeline
- Training loop with validation
- Interactive text generation
- Comprehensive evaluation metrics

## 📊 Results
- Training Loss: 0.031
- Validation Loss: 0.031
- Perplexity: 1.03
- Vocabulary Efficiency: 1.48%

## 🛠️ Usage
```bash
# Train the model
python calculate_loss.py

# Test generation
python final_resume_generator.py

# Evaluate performance
python preplexity_calculation.py
```

## 🎯 Architecture
- **Layers**: 3
- **Heads**: 4
- **Embedding Dim**: 256
- **Block Size**: 256
- **Parameters**: 15.2M
