"""Entry point: run the document ingestion pipeline end to end."""

from src.rag.ingest import run_ingestion

if __name__ == "__main__":
    run_ingestion()
