import os
import yaml
from dotenv import load_dotenv
from rag_chain import RAGChain

# Local imports
from splitters import split_documents
from data_loader import load_file, load_directory
from embeddings import load_embeddings_model
from vectorstores import build_vectorstore
from retrievers import Retriever
from generator import Generator


class StandardRAG:        
    def __init__(self, file_path: str = None):
        """
        Initialize a Standard RAG pipeline using config.yaml.
        """
        load_dotenv()

        # === Load config ===
        with open("./config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # === File handling ===
        if file_path:
            docs = load_file(file_path)
        else:
            docs = load_directory("./data")

        if not docs:
            raise ValueError("No documents found!")

        # === Splitter ===
        splitter_cfg = config["splitter"]
        chunks = split_documents(
            splitter_name=splitter_cfg["name"],
            documents=docs,
            chunk_size=splitter_cfg["chunk_size"],
            chunk_overlap=splitter_cfg["chunk_overlap"],
            separator=splitter_cfg.get("separator", None),
            token_chunk_size=splitter_cfg.get("token_chunk_size", None),
            token_chunk_overlap=splitter_cfg.get("token_chunk_overlap", None),
        )

        # === Embeddings ===
        emb_cfg = config["embeddings"]
        self.emb = load_embeddings_model(
            provider=emb_cfg["provider"],
            model_name=emb_cfg["model_name"],
        )

        # === Vectorstore ===
        vs_cfg = config["vectorstore"]
        self.vectorstore = build_vectorstore(
            name=vs_cfg["name"],
            chunks=chunks,
            embeddings_model=self.emb,
        )

        # === Generator ===
        gen_cfg = config["generator"]
        generator = Generator(
            provider=gen_cfg["provider"],
            model_name=gen_cfg["model_name"],
            max_tokens=gen_cfg["max_tokens"],
            max_retries=gen_cfg["max_retries"],
            temperature=gen_cfg["temperature"],
            timeout=gen_cfg["timeout"],
            top_p=gen_cfg["top_p"],
        )
        self.llm = generator.client

        # === Retriever ===
        retr_cfg = config["retriever"]
        self.retriever = Retriever(
            retriever_type=retr_cfg["type"],
            vectorstore=self.vectorstore,
            k=retr_cfg["k"],
        )

        # === Chain ===
        self.conversation_chain = RAGChain(
            retriever=self.retriever,
            embedding_model=self.emb,
            memory=None,  # Standard RAG → no memory
            generator=generator,
        )

    def ask(self, query: str) -> str:
        """
        Query the Standard RAG pipeline and return the response.
        """
        response = self.conversation_chain.invoke(query)
        return str(response)
