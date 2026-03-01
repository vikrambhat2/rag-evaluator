# LLM Evaluation for RAG Systems

## Why Evaluate RAG Systems?

Evaluating a RAG system is essential to ensure it produces accurate, relevant, and faithful answers. Without systematic evaluation, it is impossible to know whether changes to the retrieval pipeline, chunking strategy, or prompt template actually improve performance. Evaluation also helps identify specific failure modes such as poor retrieval, hallucination, or irrelevant answers.

## Evaluation Dimensions

RAG evaluation typically covers four key dimensions:

### Faithfulness

Faithfulness measures whether the generated answer is supported by the retrieved context. A faithful answer only contains claims that can be verified from the provided chunks. If the answer includes information not present in the context, it is considered a hallucination.

To evaluate faithfulness, an LLM judge examines each claim in the answer and checks whether it appears in the retrieved chunks. The score is the proportion of claims that are supported. A faithfulness score of 1.0 means every claim in the answer is grounded in the context.

### Answer Relevance

Answer relevance measures how well the generated answer addresses the user's question. An answer might be factually correct and faithful to the context but still fail to address what the user actually asked. For example, if a user asks "What is the context window of nomic-embed-text?" and the answer discusses embedding dimensions instead, the answer is not relevant.

Evaluation involves comparing the semantic alignment between the question and the answer. This can be done by generating potential questions from the answer and measuring their similarity to the original question, or by directly prompting an LLM judge to rate relevance.

### Context Precision

Context precision measures how many of the retrieved chunks are actually relevant to answering the query. If a system retrieves 5 chunks but only 2 contain relevant information, the context precision is low. High context precision means the retrieval system is selective and returns mostly useful chunks.

Low context precision leads to wasted context window space and can confuse the generation model with irrelevant information. It may also increase latency and cost since larger prompts require more tokens.

### Context Recall

Context recall measures how much of the information needed to answer the query was actually retrieved. Even if all retrieved chunks are relevant (high precision), important information might be missing from the retrieval results. Context recall is evaluated by comparing the retrieved context against a ground truth answer and checking whether all necessary information is present.

High recall is important for questions that require synthesizing information from multiple sources. Low recall means the retrieval system is missing relevant documents, which directly impacts answer quality.

## LLM-as-Judge Approach

The LLM-as-Judge paradigm uses a language model to evaluate the outputs of another language model. This approach is more scalable than human evaluation and more nuanced than traditional metrics like BLEU or ROUGE.

### How It Works

1. Define a clear evaluation prompt that describes the metric and scoring criteria.
2. Provide the LLM judge with the necessary inputs (query, answer, context, ground truth).
3. Ask the judge to output a structured score, typically a float between 0.0 and 1.0.
4. Parse the structured output to extract the numerical score.

### Judge Prompt Design

Effective judge prompts should:
- Clearly define what the metric measures
- Provide explicit scoring criteria (what constitutes 0.0 vs 1.0)
- Include instructions to output only structured JSON
- Avoid ambiguous language that could lead to inconsistent scoring
- Specify that the judge should evaluate based solely on the provided inputs

### Limitations of LLM Judges

- **Self-bias**: Models may rate their own outputs higher than those of other models.
- **Position bias**: Judges may favor content appearing earlier or later in the prompt.
- **Verbosity bias**: Longer answers may receive higher scores regardless of quality.
- **Inconsistency**: The same input may receive different scores across runs due to sampling randomness.

To mitigate these biases, use low temperature settings, run evaluations multiple times and average scores, and calibrate with human-labeled examples.

## Building an Evaluation Pipeline

A complete RAG evaluation pipeline includes:

1. **Test Set Generation**: Create question-answer pairs from the knowledge base. Each pair consists of a question, a ground truth answer, and optionally the source chunks. Automated generation using an LLM can scale this process.

2. **RAG Execution**: Run each test question through the RAG pipeline to get the generated answer and retrieved chunks.

3. **Metric Computation**: Apply each evaluation metric to the RAG outputs. This produces per-query scores for faithfulness, answer relevance, context precision, and context recall.

4. **Aggregation and Reporting**: Compute average scores across all test queries. Identify patterns in low-scoring queries. Flag metrics below acceptable thresholds (commonly 0.70) as areas needing improvement.

5. **Iteration**: Use evaluation results to guide improvements. Low faithfulness suggests better prompting or post-processing. Low context precision suggests retrieval tuning. Low context recall suggests re-chunking or expanding the knowledge base.

## Automated Test Set Generation

Generating test sets manually is time-consuming. An effective automated approach:

1. For each chunk in the knowledge base, prompt an LLM to generate question-answer pairs grounded in the chunk's content.
2. Deduplicate questions that are too similar using embedding similarity.
3. Validate that each ground truth answer is actually supported by the chunk.
4. Cap the test set size to keep evaluation tractable.

This approach produces test cases that cover the entire knowledge base and have verified ground truths, making them ideal for systematic evaluation.
