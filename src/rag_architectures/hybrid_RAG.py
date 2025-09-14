import os
from dotenv import load_dotenv
import yaml

from splitters import split_documents
from data_loader import load_file, load_directory
from embeddings import load_embeddings_model
from vectorstores import build_vectorstore
from retrievers import Retriever
from generator import Generator
from memory import ConversationMemory
from rag_chain import RAGChain


class HybridRAG:
    def __init__(self, config_path: str = "./config/config.yaml", file_path: str = None):
        """
        Initialize a Hybrid RAG pipeline (dense + sparse retrievers) reading all parameters from config.
        """
        load_dotenv()

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 1. Load documents
        if file_path:
            docs = load_file(file_path)
        else:
            docs = load_directory("./data")

        if not docs:
            raise ValueError("No documents found!")

        # 2. Split documents
        splitter_cfg = cfg.get("splitter", {})
        chunks = split_documents(
            documents=docs,
            splitter_name=splitter_cfg.get("name", "recursive"),
            chunk_size=splitter_cfg.get("chunk_size", 500),
            chunk_overlap=splitter_cfg.get("chunk_overlap", 50),
            separator=splitter_cfg.get("separator", "\n\n"),
            token_chunk_size=splitter_cfg.get("token_chunk_size", 256),
            token_chunk_overlap=splitter_cfg.get("token_chunk_overlap", 20),
        )

        # 3. Load embeddings
        emb_cfg = cfg.get("embeddings", {})
        self.emb = load_embeddings_model(
            provider=emb_cfg.get("provider", "huggingface"),
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        )

        # 4. Build vectorstore
        vs_cfg = cfg.get("vectorstore", {})
        vectorstore = build_vectorstore(
            name=vs_cfg.get("name", "faiss"),
            chunks=chunks,
            embeddings_model=self.emb,
        )

        # 5. Build hybrid retriever
        retr_cfg = cfg.get("retriever", {})
        self.retriever = Retriever(
            retriever_type="hybrid",
            vectorstore=vectorstore,
            docs=docs,
            k=retr_cfg.get("k", 3),
            weights= [0.6, 0.4],
        )

        # 6. Generator
        gen_cfg = cfg.get("generator", {})
        generator = Generator(
            provider=gen_cfg.get("provider", "openai"),
            model_name = gen_cfg.get("model_name", "gpt-4o-mini"),
            max_tokens=gen_cfg.get("max_tokens", 500),
            temperature=gen_cfg.get("temperature", 0.6),
            top_p=gen_cfg.get("top_p", 0.9),
            timeout=gen_cfg.get("timeout", 10),
            max_retries=gen_cfg.get("max_retries", 2),
        )
        self.llm = generator.client

        # 7. Memory (optional)
        memory = ConversationMemory()

        # 8. Create RAG chain
        self.conversation_chain = RAGChain(
            retriever=self.retriever,
            embedding_model=self.emb,
            memory=memory,
            generator=generator,
        )

    def ask(self, query: str) -> str:
        """
        Query the Hybrid RAG pipeline and return the response as a string.
        """
        response = self.conversation_chain.invoke(query)
        return str(response)
