import os
from dotenv import load_dotenv
from memory import ConversationMemory
from rag_chain import RAGChain
import yaml

# Local imports
from splitters import split_documents
from data_loader import load_file, load_directory
from embeddings import load_embeddings_model
from vectorstores import build_vectorstore
from retrievers import Retriever
from generator import Generator


class MemoryRAG:
    def __init__(self, file_path: str = None, config_path: str = "./config/config.yaml"):
        """
        Initialize a Memory-enabled RAG pipeline using config.yaml parameters.
        """
        load_dotenv()

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

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
            chunk_size=cfg["splitter"]["chunk_size"],      # Maximum chunk size for the chosen splitter
            chunk_overlap=cfg["splitter"]["chunk_overlap"],# Number of overlapping tokens/chars
        )

        # --- 3. Load embeddings ---
        self.emb = load_embeddings_model(
            provider=cfg["embeddings"]["provider"],        # Options: "huggingface", "openai", "cohere"
            model_name=cfg["embeddings"]["model_name"],
        )

        # --- 4. Build vectorstore ---
        self.vectorstore = build_vectorstore(
            name=cfg["vectorstore"]["name"],               # Options: "faiss", "chroma", "weaviate", "pinecone"
            chunks=chunks,
            embeddings_model=self.emb,
        )

        # --- 5. Generator (LLM client) ---
        gen_cfg = cfg["generator"]
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

        # --- 6. Retriever ---
        self.retriever = Retriever(
            retriever_type=cfg["retriever"].get("type", "dense"),  # "dense", "sparse", or "hybrid"
            vectorstore=self.vectorstore,
            k=cfg["retriever"]["k"],                                # Number of top documents to retrieve per query
        )

        # --- 7. Memory ---
        memory = ConversationMemory()

        # --- 8. Create RAG chain ---
        self.conversation_chain = RAGChain(
            retriever=self.retriever,
            embedding_model=self.emb,
            memory=memory,
            generator=generator,
        )

    def ask(self, query: str) -> str:
        """
        Query the Memory RAG pipeline and return the response as a string.
        """
        response = self.conversation_chain.invoke(query)
        return str(response)
