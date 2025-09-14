import os
from dotenv import load_dotenv
import yaml

from splitters import split_documents
from data_loader import load_file, load_directory
from embeddings import load_embeddings_model
from vectorstores import build_vectorstore
from retrievers import Retriever
from rerankers import RerankRetriever
from generator import Generator
from memory import ConversationMemory
from rag_chain import RAGChain

# HuggingFace reranker
from sentence_transformers import CrossEncoder


class RerankRAG:
    def __init__(self, file_path: str = None, config_path: str = "./config/config.yaml"):
        """
        Initialize a RAG pipeline with dense retrieval + reranking using config.yaml.
        """
        load_dotenv()

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            self.cfg = cfg
        # --- 1. Load documents ---
        if file_path:
            docs = load_file(file_path)
        else:
            docs = load_directory("./data")  # default folder

        if not docs:
            raise ValueError("No documents found!")

        # --- 2. Split into chunks ---
        chunks = split_documents(
            splitter_name=cfg["splitter"]["name"],          # Options: "recursive", "character", "token"
            documents=docs,
            chunk_size=cfg["splitter"]["chunk_size"],      # Max chunk size for selected splitter
            chunk_overlap=cfg["splitter"]["chunk_overlap"],# Overlap between chunks
        )

        # --- 3. Embeddings ---
        self.emb = load_embeddings_model(
            provider=cfg["embeddings"]["provider"],        # Options: "huggingface", "openai", "cohere"
            model_name=cfg["embeddings"]["model_name"],
        )

        # --- 4. Build FAISS vectorstore (dense retriever) ---
        self.vectorstore = build_vectorstore(
            name=cfg["vectorstore"]["name"],               # Options: "faiss", "chroma", "weaviate", "pinecone"
            chunks=chunks,
            embeddings_model=self.emb,
        )
        self.retriever = Retriever(
            retriever_type="dense",
            vectorstore=self.vectorstore,
            k=cfg["retriever"]["k"],                       # Number of top docs to retrieve per query
        )

        # --- 5. Reranker model ---
        self.reranker = CrossEncoder(cfg["reranker"]["model_name"])

        # --- 6. Generator (LLM client) ---
        gen_cfg = cfg["generator"]
        generator = Generator(
            provider=gen_cfg["provider"],
            model_name=gen_cfg["model_name"],
            max_tokens=gen_cfg["max_tokens"],
            max_retries=gen_cfg["max_retries"],
            temperature=gen_cfg["temperature"],
            timeout=gen_cfg["timeout"],
            top_p=gen_cfg["top_p"]
        )

        # --- 7. Memory (optional, enabled here) ---
        memory = ConversationMemory()

        # --- 8. Wrap retriever with reranking ---
        self.retrievee = RerankRetriever(self.retriever, self.reranker, self.cfg["reranker"]["top_k"])

        # --- 9. Create RAG chain ---
        self.conversation_chain = RAGChain(
            retriever=self.retriever,
            embedding_model=self.emb,
            memory=memory,
            generator=generator,
        )


    def ask(self, query: str) -> str:
        """
        Query the Rerank RAG pipeline and return response as a string.
        """
        response = self.conversation_chain.invoke(query)
        return str(response)
