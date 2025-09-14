"""
test_embeddings.py

Quick test for all embedding providers.
"""

import os
from embeddings import load_embeddings_model

def test_embeddings(provider: str, model_name: str = None):
    print(f"\n--- Testing {provider.upper()} ---")
    try:
        emb_model = load_embeddings_model(provider, model_name)
        sample_texts = [
            "The Eiffel Tower is in Paris.",
            "RAG combines retrieval and generation."
        ]
        vectors = emb_model.embed_documents(sample_texts)
        print(f"Sample vector length: {len(vectors[0])}")
        print(f"Number of vectors returned: {len(vectors)}")
    except Exception as e:
        print(f"❌ {provider} failed: {e}")


if __name__ == "__main__":
    # Providers to test
    providers = [
        {"provider": "openai", "model_name": "text-embedding-3-small"},
        {"provider": "huggingface", "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        {"provider": "cohere", "model_name": "embed-english-v3.0"},
    ]

    for p in providers:
        test_embeddings(p["provider"], p.get("model_name"))
