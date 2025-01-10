# Drug Information RAG System

A Retrieval-Augmented Generation (RAG) system for drug-related question answering, comparing multiple language models.

## Overview

This system combines document retrieval using FAISS with multiple language models (GPT-2, T5, BERT, DistilBERT, and BLOOM) to answer questions about drugs. It includes components for data processing, embedding generation, similarity search, and answer generation.

## Features

- Document chunking and preprocessing
- Semantic search using FAISS index
- Multiple model support:
  - GPT-2
  - T5
  - BERT
  - DistilBERT
  - BLOOM
- Comprehensive evaluation metrics
- Performance benchmarking

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
```

## Project Structure

```
├── data/
│   └── sample.csv        # Input drug data
├── processed_data/
│   ├── processed_data.csv
│   └── data_with_embeddings_final.parquet
├── models/
│   └── faiss_index_final.bin
└── evaluation/
    └── rag_evaluation_results.csv
```

## Usage

### 1. Data Processing

```python
# Load and process the data
df = pd.read_csv("sample.csv")
processed_df = process_data(df)
processed_df.to_csv("processed_data.csv", index=False)
```

### 2. Generate Embeddings

```python
# Generate embeddings using SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
processed_df['embeddings'] = processed_df['chunk'].apply(lambda x: model.encode(x).tolist())
```

### 3. Create FAISS Index

```python
# Convert embeddings to numpy array and create FAISS index
embeddings = np.array(processed_df['embeddings'].tolist(), dtype='float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

### 4. Query the System

```python
# Example query
query = "What is Bismuth Stibium?"
query_embedding = model.encode(query).astype('float32').reshape(1, -1)
distances, indices = index.search(query_embedding, 5)
relevant_contexts = processed_df.iloc[indices[0]]['chunk'].tolist()
```

### 5. Generate Answers

```python
# Generate answers using different models
gpt2_answer = generate_answer_with_context_gpt2(query, context)
t5_answer = generate_answer_with_context_t5(query, context)
bert_answer = generate_answer_with_context_bert(query, context)
distilbert_answer = generate_answer_with_context_distilbert(query, context)
bloom_answer = generate_answer_with_context_bloom(query, context)
```

## Evaluation

The system includes comprehensive evaluation metrics:

- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU score
- Exact match score
- F1 score
- Execution time

To run evaluation:

```python
test_data = [
    {
        'question': 'What is Bismuth Stibium?',
        'context': get_relevant_context('What is Bismuth Stibium?'),
        'reference_answer': 'Reference answer here'
    }
]

evaluation_results = evaluate_all_models(test_data, models)
```

## Model Performance

Typical performance metrics (example):
- GPT-2: Best ROUGE-1 (0.2872) and BLEU (0.3203)
- T5: Best ROUGE-2 (0.1061)
- BLOOM: Best ROUGE-L (0.2088)
- DistilBERT: Fastest execution (0.87s)

## Limitations

- Token length limitations for each model
- Context retrieval quality depends on embedding similarity
- Processing time varies significantly between models
- Memory requirements for larger datasets

## Future Improvements

1. Model fine-tuning for domain-specific knowledge
2. Improved context retrieval methods
3. Hybrid search combining semantic and keyword matching
4. Streaming response generation
5. Caching frequently asked questions

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Contact

[Your contact information]
