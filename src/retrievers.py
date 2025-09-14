import os
import requests
from typing import List, Any
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class Retriever:
    """
    Unified Retriever class supporting:
      - Dense retriever (vectorstore-based)
      - Hybrid retriever (dense + BM25 sparse)
      - Web retriever (Serper API)

    All retrievers expose the same `.invoke(query)` method.
    """

    def __init__(
        self,
        retriever_type: str,
        vectorstore=None,
        docs=None,
        k: int = 3,
        weights: List[float] = None,
    ):
        self.retriever_type = retriever_type
        self.k = k
        self.vectorstore = vectorstore
        self.weights = weights or [0.6, 0.4]

        if retriever_type == "dense":
            if vectorstore is None:
                raise ValueError("vectorstore is required for dense retriever.")
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        elif retriever_type == "hybrid":
            if vectorstore is None or docs is None:
                raise ValueError("Both vectorstore and docs are required for hybrid retriever.")
            dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            sparse_retriever = BM25Retriever.from_documents(docs)
            sparse_retriever.k = k
            self.retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=self.weights,
            )

        elif retriever_type == "web":
            self.api_key = os.getenv("SERPER_API_KEY")
            if not self.api_key:
                raise ValueError("Missing SERPER_API_KEY in environment.")
            self.endpoint = "https://google.serper.dev/search"

        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}")

    def invoke(self, query: str) -> List[Any]:
        if self.retriever_type == "dense":
            return self.retriever.invoke(query)

        elif self.retriever_type == "hybrid":
            return self.retriever.invoke(query)

        elif self.retriever_type == "web":
            headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
            payload = {"q": query, "num": self.k}
            resp = requests.post(self.endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            docs = []
            if "organic" in data:
                for item in data["organic"][: self.k]:
                    snippet = item.get("snippet", "")
                    title = item.get("title", "")
                    link = item.get("link", "")
                    text = f"{title}\n{snippet}\nSource: {link}"
                    # wrap as object with .page_content like LangChain docs
                    docs.append(type("Doc", (), {"page_content": text})())
            return docs

        else:
            raise ValueError(f"Unsupported retriever_type: {self.retriever_type}")
