"""
test_splitters.py

Test file for splitter.py to ensure all splitting strategies work correctly.
"""

from langchain.schema import Document
from splitters import split_documents

def main():
    # ------------------------------
    # Sample documents
    # ------------------------------
    docs = [
        Document(page_content="The European Parliament is the legislative branch of the European Union."),
        Document(page_content="The Eiffel Tower is located in Paris, France. It was completed in 1889 and is a world-famous landmark."),
        Document(page_content="Python is a popular programming language for AI and data science. It has extensive libraries and frameworks."),
    ]
    # ------------------------------
    # Splitters to test
    # ------------------------------
    splitters = ["recursive", "character", "token"]

    for splitter_name in splitters:
        print(f"\n--- Testing {splitter_name.upper()} Splitter ---")
        try:
            # You can pass custom parameters via kwargs if needed
            kwargs = {}
            if splitter_name == "recursive":
                kwargs = {"chunk_size": 500, "chunk_overlap": 50}
            elif splitter_name == "character":
                kwargs = {"separator": "\n\n", "chunk_size": 500, "chunk_overlap": 50}
            elif splitter_name == "token":
                kwargs = {"chunk_size": 256, "chunk_overlap": 20}

            chunks = split_documents(docs, splitter_name=splitter_name, **kwargs)
            print(f"Total chunks: {len(chunks)}")
        except Exception as e:
            print(f"❌ {splitter_name} splitter failed: {e}")


if __name__ == "__main__":
    main()
