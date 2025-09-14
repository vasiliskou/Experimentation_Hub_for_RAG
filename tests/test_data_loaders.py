# tests/test_data_loaders.py

import pytest
from pathlib import Path
from data_loader import load_file, load_directory, load_hf_dataset
from langchain.schema import Document

# -------------------------------
# File paths for testing
# -------------------------------
DATA_DIR = Path("data/")

FILE_PATHS = [
    DATA_DIR / "sample.pdf",
    DATA_DIR / "sample.txt",
    DATA_DIR / "sample.docx",
    DATA_DIR / "sample.md",
    DATA_DIR / "sample.html",
    DATA_DIR / "sample.json",
]

# -------------------------------
# Tests for individual file loaders
# -------------------------------
@pytest.mark.parametrize("file_path", FILE_PATHS)
def test_file_loader(file_path):
    """
    Test loading a single file of each supported type.
    """
    docs = load_file(str(file_path))
    assert isinstance(docs, list), f"{file_path} did not return a list"
    assert len(docs) > 0, f"No documents loaded from {file_path}"
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content.strip() != ""
        assert hasattr(doc, "metadata")
    print(f"[PASS] Loaded {len(docs)} docs from {file_path.name}")


# -------------------------------
# Test directory loader
# -------------------------------
def test_directory_loader():
    """
    Test loading all supported files from a directory.
    """
    docs = load_directory(str(DATA_DIR))
    assert isinstance(docs, list)
    assert len(docs) > 0, "No documents loaded from directory"
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content.strip() != ""
    print(f"[PASS] Loaded {len(docs)} docs from directory")


# -------------------------------
# Test HuggingFace dataset loader
# -------------------------------
def test_hf_dataset_loader():
    """
    Test loading a small subset from HuggingFace dataset.
    """
    # Limit to 3 for fast testing
    docs = load_hf_dataset(
        dataset_name="coastalcph/multi_eurlex",
        split="train",
        limit=3,
        text_field="text",
        id_field="celex_id"
    )
    assert len(docs) == 3, "HF dataset loader did not return correct number of docs"
    for doc in docs:
        assert isinstance(doc, Document)
        assert len(doc.page_content) > 0
        assert "celex_id" in doc.metadata
    print("[PASS] HuggingFace dataset loaded successfully")


# -------------------------------
# Main block to run manually
# -------------------------------
if __name__ == "__main__":
    for file_path in FILE_PATHS:
        test_file_loader(file_path)
    test_directory_loader()
    test_hf_dataset_loader()
    print("\nAll data loader tests passed!")
