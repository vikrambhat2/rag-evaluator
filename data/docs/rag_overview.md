# Retrieval-Augmented Generation (RAG) Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by combining them with external knowledge retrieval. Instead of relying solely on the model's parametric memory, RAG systems retrieve relevant documents from a knowledge base and use them as context for generating responses.

## Core Architecture

A typical RAG system consists of three main components:

1. **Document Ingestion Pipeline**: This component processes raw documents, splits them into manageable chunks, converts them into vector embeddings, and stores them in a vector database. The ingestion pipeline runs offline and prepares the knowledge base for retrieval.

2. **Retrieval Module**: When a user query arrives, the retrieval module converts it into an embedding using the same model used during ingestion. It then performs a similarity search against the vector database to find the most relevant document chunks. Common similarity metrics include cosine similarity, dot product, and Euclidean distance.

3. **Generation Module**: The retrieved chunks are combined with the original query into a prompt that is sent to the language model. The LLM generates an answer grounded in the retrieved context, reducing hallucination and improving factual accuracy.

## Why RAG Matters

Traditional LLMs suffer from several limitations that RAG addresses:

- **Knowledge Cutoff**: LLMs are trained on data up to a certain date. RAG allows them to access up-to-date information stored in the knowledge base.
- **Hallucination**: Without grounding, LLMs may generate plausible but incorrect information. RAG reduces this by providing factual context.
- **Domain Specificity**: RAG enables LLMs to answer questions about proprietary or domain-specific data that was not in the training set.
- **Transparency**: Retrieved documents can be shown to users as sources, making the system more transparent and auditable.

## RAG vs Fine-Tuning

Fine-tuning modifies the model weights to incorporate new knowledge, while RAG keeps the model unchanged and provides knowledge at inference time. RAG is preferred when knowledge changes frequently, when you need source attribution, or when you want to avoid the cost and complexity of fine-tuning. Fine-tuning is better for teaching the model new behaviors, styles, or formats.

## Common RAG Patterns

### Naive RAG
The simplest form: embed query, retrieve top-k chunks, concatenate them into a prompt, and generate. This works well for straightforward factual questions but can struggle with complex multi-hop reasoning.

### Advanced RAG
Adds pre-retrieval and post-retrieval processing steps. Pre-retrieval techniques include query rewriting, query expansion, and hypothetical document embeddings (HyDE). Post-retrieval techniques include re-ranking retrieved documents, compressing context, and filtering irrelevant chunks.

### Modular RAG
A flexible architecture where components can be swapped or combined. For example, you might use a sparse retriever (BM25) alongside a dense retriever and fuse their results. Modular RAG supports iterative retrieval, where the model can issue multiple retrieval calls to refine its answer.

## Key Metrics for RAG Evaluation

Evaluating a RAG system requires assessing both retrieval quality and generation quality:

- **Context Precision**: How many of the retrieved chunks are actually relevant to the query?
- **Context Recall**: How much of the relevant information in the knowledge base was retrieved?
- **Faithfulness**: Is the generated answer supported by the retrieved context, or does it contain hallucinated information?
- **Answer Relevance**: Does the generated answer actually address the user's question?

These metrics together provide a comprehensive view of RAG system performance and help identify weak points in the pipeline.
