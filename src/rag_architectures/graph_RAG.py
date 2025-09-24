import os
import yaml
from dotenv import load_dotenv

from data_loader import load_file, load_directory
from splitters import split_documents
from memory import ConversationMemory
from generator import Generator
from graphs import Graph

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain


class GraphRAG:
    def __init__(self, file_path: str = None, config_path: str = "./config/config.yaml"):
        """
        Initialize a Graph RAG pipeline using config.yaml parameters.
        """
        load_dotenv()

        # --- 1. Load configuration ---
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # --- 2. Load documents ---
        if file_path:
            docs = load_file(file_path)
        else:
            docs = load_directory(cfg.get("data", {}).get("path", "./data"))

        if not docs:
            raise ValueError("No documents found!")

        chunks = split_documents(
            splitter_name=cfg["splitter"]["name"],
            documents=docs,
            chunk_size=cfg["splitter"]["chunk_size"],
            chunk_overlap=0,
        )

        # --- 3. Initialize Generator (LLM client) ---
        generator = Generator(
            provider="openai",
            model_name="gpt-4o-mini",
            max_tokens=None,
            max_retries=2,
            temperature=0.3,
            timeout=10,
            top_p=0.9,
        )
        self.llm = generator.client

        # --- 4. Convert documents into graph representation ---
        self.graph = Graph(llm=self.llm, chunks=chunks).graph

        # --- 5. Build Graph QA chain ---
        self.chain = GraphQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
        )

    def ask(self, query: str) -> str:
        """
        Query the Graph RAG pipeline and return the response as a string.
        """
        response = self.chain.run(query)
        return response
