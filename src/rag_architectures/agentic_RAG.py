import os
from dotenv import load_dotenv

from retrievers import Retriever
from splitters import split_documents
from data_loader import load_file, load_directory
from embeddings import load_embeddings_model
from vectorstores import build_vectorstore
from generator import Generator
from rag_chain import RAGChain
from memory import ConversationMemory
from agents import AgentWorkflow


class AgenticRAG:
    def __init__(self, file_path: str = None, config: dict = None):
        """
        Agentic RAG pipeline using local vector DB, web search, or history.
        Configuration is read from a YAML/dict (with defaults if not provided).
        """
        load_dotenv()

        # Load config sections with defaults
        splitter_cfg = config.get("splitter", {}) if config else {}
        emb_cfg = config.get("embeddings", {}) if config else {}
        retr_cfg = config.get("retriever", {}) if config else {}
        vec_cfg = config.get("vectorstore", {}) if config else {}
        gen_cfg = config.get("generator", {}) if config else {}
        rerank_cfg = config.get("reranker", {}) if config else {}

        # === Load documents ===
        if file_path:
            docs = load_file(file_path)
        else:
            docs = load_directory("./data")

        if not docs:
            raise ValueError("No documents found!")

        # === Split documents ===
        chunks = split_documents(
            splitter_name=splitter_cfg.get("name", "recursive"),
            documents=docs,
            chunk_size=splitter_cfg.get("chunk_size", 500),
            chunk_overlap=splitter_cfg.get("chunk_overlap", 50),
        )

        # === Embeddings ===
        self.emb = load_embeddings_model(
            provider=emb_cfg.get("provider", "huggingface"),
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        )

        # === Vectorstore ===
        vectorstore = build_vectorstore(
            name=vec_cfg.get("name", "faiss"),
            chunks=chunks,
            embeddings_model=self.emb,
            persist_directory=vec_cfg.get("persist_directory", None),
            index_name=vec_cfg.get("index_name", None),
            embeddings_dim=vec_cfg.get("embeddings_dim", None),
            similarity_metric=vec_cfg.get("similarity_metric", "cosine"),
        )

        # === Local retriever ===
        self.local_retriever = vectorstore.as_retriever(
            search_kwargs={"k": retr_cfg.get("k", 3)}  # top-k docs
        )

        # === Web retriever ===
        self.web_retriever = Retriever(
            retriever_type="web", k=retr_cfg.get("k", 3)
        )

        # === Generator ===
        self.generator = Generator(
            provider=gen_cfg.get("provider", "openai"),
            model_name=gen_cfg.get("model_name", "gpt-4o-mini"),
            max_tokens=gen_cfg.get("max_tokens", 500),   # aligned with YAML
            temperature=gen_cfg.get("temperature", 0.6),
            max_retries=gen_cfg.get("max_retries", 2),
            timeout=gen_cfg.get("timeout", 10),
            top_p=gen_cfg.get("top_p", 0.9),
        )

        # === Agent workflow ===
        self.workflow = AgentWorkflow()

        # === Memory ===
        self.memory = ConversationMemory()

        # === RAG chain (retriever chosen dynamically per query) ===
        self.conversation_chain = RAGChain(
            retriever=None,
            embedding_model=self.emb,
            memory=self.memory,
            generator=self.generator,
        )

    def ask(self, query: str) -> str:
        """
        Decide whether to use local retriever, web retriever, or history,
        then run RAG pipeline and return answer.
        """
        # 1. Planner decision
        result = self.workflow.run(query)
        source = result.get("source", "local")
        refined_query = result.get("query", query)

        # 2. Choose retriever & fetch docs
        if source == "web":
            self.retriever = self.web_retriever
            docs = self.retriever.invoke(refined_query)
        elif source == "history" and self.memory:
            docs = [{"page_content": "Answer based on conversation history not context."}]
        else:
            self.retriever = self.local_retriever
            docs = self.retriever.invoke(refined_query)

        # 3. Build prompt
        prompt = self.conversation_chain._build_prompt(refined_query, docs)

        # 4. Generate answer
        answer = self.generator.generate(
            system_prompt=self.conversation_chain.system_prompt,
            user_prompt=prompt,
        )

        # 5. Update memory
        if self.memory:
            self.memory.add_message("user", refined_query)
            self.memory.add_message("assistant", answer)

        return answer
