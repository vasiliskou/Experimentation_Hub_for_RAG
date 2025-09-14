import os
import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)

from langchain.schema import Document


def load_pdf(path: str) -> List[Document]:
    """Load a PDF file into LangChain Documents."""
    loader = PyPDFLoader(path)
    return loader.load()


def load_txt(path: str, encoding: str = "utf-8") -> List[Document]:
    """Load a plain text file."""
    loader = TextLoader(path, encoding=encoding)
    return loader.load()


def load_docx(path: str) -> List[Document]:
    """Load a Word document (.docx)."""
    loader = UnstructuredWordDocumentLoader(path)
    return loader.load()


def load_markdown(path: str) -> List[Document]:
    """Load a Markdown file."""
    loader = UnstructuredMarkdownLoader(path)
    return loader.load()


def load_html(path: str) -> List[Document]:
    """Load an HTML file."""
    loader = UnstructuredHTMLLoader(path)
    return loader.load()


def load_json(path: str, text_field: str = None) -> List[Document]:
    """
    Load JSON file into Documents.
    If text_field is provided, use that field for content.
    Otherwise, dumps entire JSON entry as string.
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]  # wrap single dict into list

    for entry in data:
        if text_field and text_field in entry:
            content = entry[text_field]
        else:
            content = json.dumps(entry, ensure_ascii=False)
        docs.append(Document(page_content=content, metadata={"source": path}))
    return docs


def load_file(path: str) -> List[Document]:
    """Generic loader that dispatches based on file extension."""
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_txt(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in [".md", ".markdown"]:
        return load_markdown(path)
    elif ext in [".html", ".htm"]:
        return load_html(path)
    elif ext == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_directory(directory: str, recursive: bool = True) -> List[Document]:
    """
    Load all supported files from a directory.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                docs = load_file(path)
                documents.extend(docs)
            except ValueError:
                # Skip unsupported file types
                continue
        if not recursive:
            break
    return documents

def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    limit: int = None,
    text_field: str = "text",
    id_field: str = "id",
) -> list:
    """
    Load a HuggingFace dataset and wrap entries into LangChain Document objects.

    Args:
        dataset_name (str): Name of the dataset on HuggingFace hub.
        split (str): Split to load (e.g., "train", "test").
        limit (int, optional): Limit number of samples (download only part).
        text_field (str): Column containing the main text.
        id_field (str): Column to use as unique identifier (or auto-generate).

    Returns:
        list[Document]: List of LangChain Document objects.
    """
    from datasets import load_dataset
    from langchain.schema import Document

    # Use HuggingFace slicing if limit is set
    if limit:
        split = f"{split}[:{limit}]"

    dataset = load_dataset(dataset_name, split=split)

    documents = []
    for i, entry in enumerate(dataset):
        # Check for the text field
        if text_field not in entry:
            raise KeyError(
                f"Text field '{text_field}' not found in dataset columns: {dataset.column_names}"
            )

        # Use id_field if present, otherwise auto-generate
        if id_field in entry:
            doc_id = entry[id_field]
        else:
            doc_id = f"{dataset_name}_{split}_{i}"

        documents.append(
            Document(page_content=entry[text_field], metadata={id_field: doc_id})
        )

    return documents



