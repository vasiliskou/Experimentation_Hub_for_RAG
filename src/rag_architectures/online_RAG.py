import os
from dotenv import load_dotenv

from retrievers import Retriever
from generator import Generator
from memory import ConversationMemory
from rag_chain import RAGChain


class OnlineRAG:
    def __init__(self, config: dict = None):
        """
        Online RAG pipeline using web search via the unified Retriever class.
        Configuration can be passed as a dictionary; defaults are used if not provided.
        """
        load_dotenv()

        # Load parameters from config or use defaults
        retr_cfg = config.get("retriever", {}) if config else {}
        gen_cfg = config.get("generator", {}) if config else {}

        # 1. Web retriever
        self.retriever = Retriever(
            retriever_type="web",
            k=retr_cfg.get("k", 5),  # Number of results to retrieve from web
        )

        # 2. Generator (LLM client)
        self.generator = Generator(
            provider=gen_cfg.get("provider", "openai"),
            model_name = gen_cfg.get("model_name", "gpt-4o-mini"),
            max_tokens=gen_cfg.get("max_tokens", 500),
            max_retries=gen_cfg.get("max_retries", 2),
            temperature=gen_cfg.get("temperature", 0.6),
            timeout=gen_cfg.get("timeout", 10),
            top_p=gen_cfg.get("top_p", 0.9),
        )

        # 3. Create RAG chain (no embeddings, memory optional)
        self.conversation_chain = RAGChain(
            retriever=self.retriever,
            embedding_model=None,
            memory=None,  # Can be replaced with ConversationMemory() if needed
            generator=self.generator,
        )

    def ask(self, query: str) -> str:
        """
        Query the Online RAG pipeline and return response.
        """
        # Retrieve docs using web retriever
        docs = self.retriever.invoke(query)

        # Build prompt manually since no embedding model is used
        prompt = self.conversation_chain._build_prompt(query, docs)

        # Generate answer
        answer = self.generator.generate(
            system_prompt=self.conversation_chain.system_prompt,
            user_prompt=prompt,
        )

        # Update memory if enabled
        if self.conversation_chain.memory:
            self.conversation_chain.memory.add_message("user", query)
            self.conversation_chain.memory.add_message("assistant", answer)

        return answer
