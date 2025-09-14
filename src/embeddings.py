"""
embeddings.py

Unified embedding model loader for vectorstores.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_embeddings_model(provider: str, model_name: str = None):
    """
    Returns an embedding model instance based on provider and optional model name.

    Supported providers:
      - openai
      - huggingface
      - cohere
    """
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )

    elif provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=model_name or "embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

