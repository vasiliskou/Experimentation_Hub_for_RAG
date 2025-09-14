class RerankRetriever:
    def __init__(self, retriever, reranker, top_k):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k

    def invoke(self, query):
        # 1. Get initial retrieved docs
        docs = self.retriever.invoke(query)
        if not docs:
            return []

        # 2. Score documents with reranker
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)

        # 3. Sort by score (descending) and keep top-k
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:self.top_k]]
