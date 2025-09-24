import gradio as gr
from dotenv import load_dotenv
import os

# Import different RAG classes
from rag_architectures.standard_RAG import StandardRAG
from rag_architectures.standard_RAG_with_memory import MemoryRAG
from rag_architectures.hybrid_RAG import HybridRAG
from rag_architectures.rerank_RAG import RerankRAG
from rag_architectures.agentic_RAG import AgenticRAG
from rag_architectures.online_RAG import OnlineRAG
from rag_architectures.graph_RAG import GraphRAG

# Load environment variables
load_dotenv()

# Store initialized RAGs to avoid reloading every time
rag_instances = {}
rag_files = {}  # keep track of last file per architecture


def get_rag_instance(arch: str, file_path: str = None):
    """
    Return the RAG instance based on architecture selection.
    Reuse instance if available, only reinitialize if a *new* file is uploaded.
    """
    last_file = rag_files.get(arch)

    # If no instance yet OR a new file has been uploaded → reinitialize
    if arch not in rag_instances or (file_path and file_path != last_file):
        if arch == "Standard RAG":
            rag_instances[arch] = StandardRAG(file_path=file_path)
        elif arch == "Standard RAG + Memory":
            rag_instances[arch] = MemoryRAG(file_path=file_path)
        elif arch == "Hybrid RAG":
            rag_instances[arch] = HybridRAG(file_path=file_path)
        elif arch == "Rerank RAG":
            rag_instances[arch] = RerankRAG(file_path=file_path)
        elif arch == "Online RAG":  # Online RAG ignores file_path
            rag_instances[arch] = OnlineRAG()
        elif arch == "Agentic RAG":
            rag_instances[arch] = AgenticRAG(file_path=file_path)
        elif arch == "Graph RAG":
            rag_instances[arch] = GraphRAG(file_path=file_path)
        else:
            raise ValueError(f"Unknown RAG architecture: {arch}")

        # update last used file for this architecture
        rag_files[arch] = file_path

    return rag_instances[arch]


def chat_with_rag(message, history, architecture, file_path=None):
    """Ask the selected RAG system and return only the answer text."""
    rag = get_rag_instance(architecture, file_path)
    response = rag.ask(message)
    return str(response)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Experimentation Hub for RAG\nChoose your architecture and ask questions!")

    with gr.Row(equal_height=True):
        arch_selector = gr.Dropdown(
            choices=[
                "Standard RAG",
                "Standard RAG + Memory",
                "Hybrid RAG",
                "Rerank RAG",
                "Online RAG",
                "Agentic RAG",
                "Graph RAG"
            ],
            value="Standard RAG",
            label="Select RAG Architecture",
        )

        file_upload = gr.File(
            label="Upload your document",
            type="filepath",
            scale=1,
            height=120,
        )

    chatbot = gr.Chatbot(height=376)
    msg = gr.Textbox(placeholder="Ask me something...", label="Your Question")

    def respond(user_message, chat_history, architecture, file_path):
        bot_message = chat_with_rag(user_message, chat_history, architecture, file_path)
        chat_history.append((user_message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot, arch_selector, file_upload], [msg, chatbot])

demo.launch()
