# Drug Information RAG System

A Retrieval-Augmented Generation (RAG) system for drug-related question answering, comparing multiple language models including state-of-the-art models like LLAMA-2 and Gemini.

## Overview

This system combines document retrieval using FAISS with multiple language models to answer questions about drugs. It includes comprehensive evaluation metrics and visualization capabilities.

## Features

### Models Supported
- GPT-2
- T5 (Large)
- BiomedBERT (Microsoft BiomedNLP)
- BERT (Large uncased)
- DistilBERT
- DistilGPT2
- ELECTRA
- BLOOM
- LLAMA-2
- Gemini

### Components
- Document chunking and preprocessing
- Semantic search using FAISS
- Multiple model comparison
- Comprehensive evaluation metrics
- Performance visualization
- API integration (Replicate API for LLAMA-2, Google AI for Gemini)

## Requirements

```bash
pip install faiss-gpu
pip install sentence-transformers
pip install rouge-score
pip install torch
pip install transformers
pip install nltk
pip install pandas
pip install numpy
pip install matplotlib
pip install google-generativeai
pip install replicate
```

## API Keys Required
- Google AI API key for Gemini
- Replicate API token for LLAMA-2

## Project Structure

```
├── data/
│   └── sample.csv        # Input drug data
├── processed_data/
│   ├── processed_data.csv
│   └── data_with_embeddings_final.parquet
├── models/
│   └── faiss_index_final.bin
├── evaluation/
│   ├── rag_evaluation_results.csv
│   └── evaluation_graphs/
│       ├── rouge1_comparison.png
│       ├── rouge2_comparison.png
│       ├── rougeL_comparison.png
│       ├── bleu_comparison.png
│       ├── f1_comparison.png
│       ├── cosine_similarity_comparison.png
│       └── execution_time_comparison.png
```

## Evaluation Metrics

The system now includes enhanced evaluation metrics:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU score
- Exact match score
- F1 score
- Cosine similarity
- Execution time

## Usage

### 1. Set up API Keys

```python
# For Gemini
os.environ["GOOGLE_API_KEY"] = "your_google_api_key"

# For LLAMA-2
os.environ["REPLICATE_API_TOKEN"] = "your_replicate_api_token"
```

### 2. Initialize Models

```python
# Import required model functions
from transformers import (
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    BertForQuestionAnswering,
    DistilBertForQuestionAnswering,
    ElectraForQuestionAnswering
)
```

### 3. Query the System

```python
# Example query
query = "What is Bismuth Stibium?"
context = get_relevant_context(query)

# Generate answers using different models
t5_answer = generate_answer_with_context_t5(query, context)
llama2_answer = generate_answer_with_context_LAMA_2(query, context)
gemini_answer = generate_answer_with_context_gemini(query, context)
```

### 4. Run Evaluation

```python
test_data = [
    {
        'question': 'What is the Active Ingredients of Bismuth Stibium?',
        'context': actual_context,
        'reference_answer': 'Active Ingredients: [reference answer]'
    }
]

results = evaluate_all_models(test_data, models)
```

### 5. Generate Visualizations

```python
# Plot comparison graphs
for metric in metrics:
    plot_metric_comparison(df_results, metric, output_directory)
```

## Visualization

The system automatically generates comparison graphs for:
- ROUGE scores comparison
- BLEU score comparison
- F1 score comparison
- Cosine similarity comparison
- Execution time comparison

Graphs are saved in the `evaluation_graphs` directory.

## Model Performance Considerations

- LLAMA-2 and Gemini typically provide more detailed and accurate responses
- T5-Large shows good performance for specific medical terminology
- BERT-based models are faster but may provide shorter answers
- Consider execution time vs accuracy tradeoffs when choosing models

## Limitations

- API rate limits for Gemini and LLAMA-2
- Token length limitations for each model
- Context retrieval quality depends on embedding similarity
- Memory requirements for larger models
- API costs for commercial usage

## Future Improvements

1. Model fine-tuning for medical domain
2. Response verification against medical databases
3. Multi-language support
4. Automated model selection based on query type
5. Batch processing capabilities
6. Response confidence scoring

