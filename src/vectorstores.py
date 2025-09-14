"""
vectorstores.py

Build and manage vectorstores from pre-split chunks and pre-initialized embeddings.
Supports: FAISS, Chroma, Weaviate, Pinecone
"""

import os
from dotenv import load_dotenv
import os

# Load environment variables (PINECONE_API_KEY etc.)
load_dotenv()

def build_vectorstore(name: str, chunks, embeddings_model, **kwargs):
    name = name.lower()

    if name == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(chunks, embeddings_model)

    elif name == "chroma":
        from langchain_community.vectorstores import Chroma
        persist_dir = kwargs.get("persist_directory")
        return Chroma.from_documents(chunks, embeddings_model, persist_directory=persist_dir)

    elif name == "weaviate":
        import weaviate
        from langchain_weaviate.vectorstores import WeaviateVectorStore
        client = weaviate.connect_to_local()
        index_name = kwargs.get("index_name", "LangChain")
        return WeaviateVectorStore.from_documents(chunks, embeddings_model, client=client, index_name=index_name)

    elif name == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        index_name = kwargs.get("index_name", "default-index")
        embeddings_dimension = kwargs.get("ebeddings_dim", 384)
        similarity_metric = kwargs.get("similarity_metric", "cosine")

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=embeddings_dimension,
                metric=similarity_metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings_model)
        vector_store.add_documents(chunks)
        return vector_store

    else:
        raise ValueError(f"Unsupported backend: {name}")

