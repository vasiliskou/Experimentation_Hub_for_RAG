from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

class Graph:
    def __init__(self, llm, chunks):
        # --- Convert chunks into graph documents ---
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(chunks)

        # --- Initialize empty graph ---
        self.graph = NetworkxEntityGraph()

        # --- Add nodes and edges from all graph_documents ---
        for gdoc in graph_documents:
            for node in gdoc.nodes:
                self.graph.add_node(node.id)
            for edge in gdoc.relationships:
                self.graph._graph.add_edge(edge.source.id, edge.target.id, relation=edge.type)
