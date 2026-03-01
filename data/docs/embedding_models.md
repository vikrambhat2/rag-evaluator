# Embedding Models for RAG Systems

## What Are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning. Two pieces of text with similar meanings will have embeddings that are close together in the vector space, even if they use different words. This property makes embeddings ideal for semantic search in RAG systems.

## How Embedding Models Work

Embedding models are typically based on transformer architectures. They process input text through multiple layers of self-attention and feed-forward networks, producing a fixed-dimensional vector that represents the input. The most common approach uses the [CLS] token output or mean pooling over all token outputs.

Training typically involves contrastive learning, where the model learns to place similar texts close together and dissimilar texts far apart. Datasets for training include natural language inference pairs, question-answer pairs, and paraphrase collections.

## Popular Open-Source Embedding Models

### Nomic Embed Text

Nomic-embed-text is an open-source embedding model that produces 768-dimensional vectors with a context window of 8192 tokens. It is available through Ollama and can be run entirely locally. It achieves competitive performance on the MTEB benchmark while being lightweight enough to run on consumer hardware.

Nomic-embed-text supports task-prefixed embeddings. For search retrieval, queries should be prefixed with "search_query:" and documents with "search_document:". This prefix mechanism helps the model distinguish between short queries and longer document passages, improving retrieval accuracy.

### Sentence Transformers (all-MiniLM-L6-v2)

This is one of the most widely used embedding models. It produces 384-dimensional vectors and has a context window of 256 tokens. Despite its small size, it performs well for many use cases. Its main limitation is the short context window, which requires careful chunking to avoid truncation.

### BGE (BAAI General Embeddings)

The BGE family of models from BAAI ranges from small (33M parameters) to large (335M parameters). BGE-large-en-v1.5 produces 1024-dimensional vectors and achieves state-of-the-art results on several benchmarks. BGE models also support instruction-prefixed queries for improved retrieval.

### E5 Models

E5 models from Microsoft use a simple prompt-based approach where queries are prefixed with "query:" and passages with "passage:". The E5-large-v2 model produces 1024-dimensional vectors and performs well across diverse tasks.

## Proprietary Embedding Models

OpenAI offers text-embedding-3-small (1536 dimensions) and text-embedding-3-large (3072 dimensions). Cohere provides embed-v3 with support for multiple languages. Google offers text-embedding-004. These models often achieve top benchmark scores but require API calls and incur costs.

## Choosing an Embedding Model

Key factors for selection:

- **Dimensionality**: Higher dimensions capture more nuance but require more storage and compute. 768 dimensions is a good balance for most applications.
- **Context window**: Must be large enough for your chunk sizes. Models with 8192-token windows like nomic-embed-text are versatile.
- **Performance**: Evaluate on the MTEB (Massive Text Embedding Benchmark) leaderboard, but also test on your specific data.
- **Deployment**: For privacy or latency requirements, local models (via Ollama or sentence-transformers) are preferable to API-based solutions.
- **Cost**: Local models have upfront compute costs but no per-query charges. API models have zero setup but ongoing costs.

## Embedding Best Practices

1. **Use the same model for queries and documents**: Mismatched embedding models produce vectors in different spaces that cannot be meaningfully compared.
2. **Normalize vectors**: Many similarity metrics assume unit-length vectors. Normalizing embeddings before storage ensures consistent similarity scores.
3. **Batch processing**: When embedding large document collections, process in batches to manage memory and take advantage of GPU parallelism.
4. **Cache embeddings**: Store computed embeddings rather than recomputing them. This saves significant time when the document collection is static.
5. **Monitor drift**: If your documents change over time, periodically re-embed to ensure the vector space stays current.

## Evaluation of Embedding Quality

The quality of embeddings can be evaluated through:

- **Retrieval accuracy**: Given a known query-document pair, does the embedding model rank the correct document highly?
- **Clustering quality**: Do embeddings of similar documents form distinct clusters?
- **Downstream task performance**: Ultimately, embedding quality is measured by the end-to-end RAG system performance on metrics like answer accuracy and faithfulness.
