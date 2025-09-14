"""
splitters.py

Utility function to split documents into smaller chunks for embeddings and vectorstores.
Supports multiple splitting strategies.
"""

from typing import List
from langchain.schema import Document


def split_documents(
    documents: List[Document],
    splitter_name: str = "recursive",
    chunk_size: int = None,
    chunk_overlap: int = None,
    separator: str = None,
    token_chunk_size: int = None,
    token_chunk_overlap: int = None,
) -> List[Document]:
    """
    Split documents using the selected splitter.

    Args:
        documents: List of Document objects to split.
        splitter_name: Which splitter to use: "recursive", "character", "token".
        chunk_size: Maximum chunk size (characters or tokens depending on splitter).
        chunk_overlap: Number of overlapping characters/tokens (recursive & character).
        separator: Separator string for "character" splitter.
            Common options:
              - "\n\n" (paragraphs) recommended default
              - "\n"   (single line)
              - " "    (spaces, word-level split)
              - ""     (character-level split)
        token_chunk_size: Maximum number of tokens per chunk (only for "token").
        token_chunk_overlap: Overlap in tokens between chunks (only for "token").

    Returns:
        List[Document]: Split documents.
    """
    splitter_name = splitter_name.lower()

    if splitter_name == "recursive":
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        chunk_size = chunk_size or 1000
        chunk_overlap = chunk_overlap or 200
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif splitter_name == "character":
        from langchain.text_splitter import CharacterTextSplitter

        chunk_size = chunk_size or 1000
        chunk_overlap = chunk_overlap or 0
        separator = separator or "\n\n"
        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif splitter_name == "token":
        from langchain.text_splitter import TokenTextSplitter

        token_chunk_size = token_chunk_size or 256
        token_chunk_overlap = token_chunk_overlap or 20
        splitter = TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=token_chunk_overlap,
        )

    else:
        raise ValueError(f"Unsupported splitter: {splitter_name}")

    return splitter.split_documents(documents)
