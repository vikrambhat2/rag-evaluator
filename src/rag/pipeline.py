import logging
from pathlib import Path

import chromadb
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "rag_docs"

ANSWER_PROMPT = """You are a helpful assistant. Answer the question based only on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""


class RAGPipeline:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.2", temperature=0.1)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def query(self, question: str) -> tuple[str, list[str]]:
        """Run a RAG query: retrieve top-k chunks and generate an answer.

        Returns:
            Tuple of (answer_text, list_of_chunk_texts)
        """
        # Retrieve
        query_embedding = self.embeddings.embed_query(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
        )

        chunks = results["documents"][0] if results["documents"] else []
        logger.info(f"Retrieved {len(chunks)} chunks for query: {question[:50]}...")

        # Generate
        context = "\n\n---\n\n".join(chunks)
        prompt = ANSWER_PROMPT.format(context=context, query=question)
        response = self.llm.invoke(prompt)
        answer = response.content.strip()

        return answer, chunks
