"""
test_vectorstores.py

Quick test for all vectorstores in vectorstores.py.
Builds small indexes and checks retrieval.
"""

import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstores import build_vectorstore

# Load environment variables (PINECONE_API_KEY etc.)
load_dotenv()

def main():
    # ------------------------------
    # Test Data
    # ------------------------------
    docs = [
        Document(page_content="The European Parliament is the legislative branch of the European Union."),
        Document(page_content="The Eiffel Tower is located in Paris, France."),
        Document(page_content="Python is a popular programming language for AI and data science."),
    ]

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------
    # Vectorstores to test
    # ------------------------------
    backends = [
        {"name": "faiss"},
        {"name": "chroma", "persist_directory": "./chroma_test"},
        {"name": "weaviate", "index_name": "TestIndex"},
        {"name": "pinecone", "index_name": "test-index", "embeddings_dim": 384, "similarity_metric": "cosine"},
    ]

    # ------------------------------
    # Run tests
    # ------------------------------
    query = "Where is the Eiffel Tower?"
    for backend in backends:
        name = backend["name"]
        kwargs = {k: v for k, v in backend.items() if k != "name"}  # remove "name"
        print(f"\n--- Testing {name.upper()} ---")

        try:
            vs = build_vectorstore(name=name, chunks=docs, embeddings_model=embeddings_model, **kwargs)
            results = vs.similarity_search(query, k=2)

            print(f"Query: {query}")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.page_content[:80]}...")

        except Exception as e:
            print(f"❌ {name} failed: {e}")

if __name__ == "__main__":
    main()
