from typing import List, Any, Optional
from memory import ConversationMemory


class RAGChain:
    def __init__(self, retriever, embedding_model, generator, memory: Optional[ConversationMemory] = None):
        """
        Custom RAG pipeline.

        Args:
            retriever: Vectorstore retriever (must implement .get_relevant_documents(query)).
            embedding_model: Embedding model used for vectorization (not used directly here).
            generator: Your Generator instance (must implement generate(system_prompt, user_prompt)).
            memory: Optional ConversationMemory instance.
        """
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.generator = generator
        self.memory = memory

        self.system_prompt = "You are a helpful assistant that answers questions."

    def _build_prompt(self, query: str, docs: List[Any]) -> str:
        """
        Build the final prompt with optional history + retrieved docs + new query.
        """
        history_text = self.memory.format_history() if self.memory else ""
        docs_text = "\n\n".join(
            [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs]
        )

        prompt = f"""
    You are a knowledgeable and reliable assistant. Your goal is to provide accurate, clear, and concise answers
    to the user's question using the retrieved documents. If the retrieved context does not contain the answer,
    you must say you don’t know rather than making something up.

    ### Conversation History
    {history_text if history_text else "(no previous conversation)"}

    ### Retrieved Context
    {docs_text if docs_text else "(no relevant documents found)"}

    ### Task
    1. Read the retrieved context carefully.
    2. If the context contains the answer, respond with a helpful explanation.
    3. If the context is unclear or missing information, acknowledge that and do not fabricate facts.
    4. Keep the answer grounded in the context. If you add general knowledge, clearly separate it from the retrieved evidence.

    ### User Question
    {query}

    ### Final Answer
    """
        return prompt.strip()


    def invoke(self, query: str) -> str:
        """
        Run the RAG pipeline: retrieve → build prompt → generate answer → (optionally) update memory.
        """
        # 1. Retrieve relevant documents
        docs = self.retriever.invoke(query)

        # 2. Build the prompt text
        prompt_text = self._build_prompt(query, docs)


        # 3. Call the Generator
        answer = self.generator.generate(
            system_prompt=self.system_prompt,
            user_prompt=prompt_text
        )

        # 4. Update memory
        if self.memory:
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", answer)

        return answer
