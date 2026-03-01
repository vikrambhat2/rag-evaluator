import json
import logging
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
TEST_SET_PATH = PROJECT_ROOT / "data" / "test_set.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "rag_docs"

QA_GENERATION_PROMPT = (
    "Given this document excerpt, generate 2 question-answer pairs. "
    "The answer must be directly supported by the text. "
    "Return only JSON, no explanation: "
    "[{\"query\": \"...\", \"ground_truth\": \"...\"}]\n"
    "Document: {chunk_text}"
)


def load_and_chunk_docs() -> list:
    """Load markdown docs from data/docs/ and chunk them."""
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def store_in_chroma(chunks: list) -> chromadb.Collection:
    """Embed chunks with nomic-embed-text and store in ChromaDB."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if present to allow re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        vectors = embeddings.embed_documents(texts)

        collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Stored batch {i // batch_size + 1} ({len(batch)} chunks)")

    logger.info(f"Total chunks in collection: {collection.count()}")
    return collection


def generate_qa_pairs(chunks: list, max_pairs: int = 15) -> list[dict]:
    """Generate QA pairs from chunks using ChatOllama, deduplicate, and cap."""
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    all_pairs: list[dict] = []

    for idx, chunk in enumerate(chunks):
        if len(all_pairs) >= max_pairs:
            break

        prompt = QA_GENERATION_PROMPT.format(chunk_text=chunk.page_content)
        logger.info(f"Generating QA pairs for chunk {idx + 1}/{len(chunks)}")

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            # Try to extract JSON from the response
            # Handle cases where model wraps JSON in markdown code blocks
            if "```" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                if start != -1 and end > start:
                    content = content[start:end]

            pairs = json.loads(content)
            if isinstance(pairs, list):
                for pair in pairs:
                    if "query" in pair and "ground_truth" in pair:
                        all_pairs.append(pair)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse QA pairs for chunk {idx}: {e}")
            continue

    # Deduplicate by query embedding similarity
    if len(all_pairs) > 1:
        all_pairs = _deduplicate_by_similarity(all_pairs, embeddings)

    # Cap at max_pairs
    all_pairs = all_pairs[:max_pairs]
    return all_pairs


def _deduplicate_by_similarity(
    pairs: list[dict], embeddings: OllamaEmbeddings, threshold: float = 0.90
) -> list[dict]:
    """Remove pairs with queries that are too similar to each other."""
    queries = [p["query"] for p in pairs]
    vectors = embeddings.embed_documents(queries)

    keep = [True] * len(pairs)
    for i in range(len(vectors)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(vectors)):
            if not keep[j]:
                continue
            sim = _cosine_similarity(vectors[i], vectors[j])
            if sim > threshold:
                keep[j] = False

    deduped = [p for p, k in zip(pairs, keep) if k]
    removed = len(pairs) - len(deduped)
    if removed:
        logger.info(f"Removed {removed} duplicate QA pairs by similarity")
    return deduped


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def save_test_set(pairs: list[dict]) -> None:
    """Save QA pairs to data/test_set.json."""
    os.makedirs(TEST_SET_PATH.parent, exist_ok=True)
    with open(TEST_SET_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved {len(pairs)} test cases to {TEST_SET_PATH}")


def run_ingestion() -> None:
    """Run the full ingestion pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load and chunk documents
    print("\n[1/3] Loading and chunking documents...")
    chunks = load_and_chunk_docs()
    print(f"  -> {len(chunks)} chunks created")

    # Step 2: Store in ChromaDB
    print("\n[2/3] Embedding and storing in ChromaDB...")
    collection = store_in_chroma(chunks)
    print(f"  -> {collection.count()} chunks stored in collection '{COLLECTION_NAME}'")

    # Step 3: Generate QA pairs
    print("\n[3/3] Generating QA test set from chunks...")
    pairs = generate_qa_pairs(chunks)
    save_test_set(pairs)
    print(f"  -> {len(pairs)} QA pairs saved to {TEST_SET_PATH}")

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
