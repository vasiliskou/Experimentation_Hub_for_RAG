import os
import time
import csv
from dotenv import load_dotenv

from splitters import split_documents
from data_loader import load_file
from embeddings import load_embeddings_model
from vectorstores import build_vectorstore
from retrievers import Retriever

load_dotenv()

EXPERIMENTS_DIR = os.path.dirname(__file__)
OUTPUT_CSV = os.path.join(EXPERIMENTS_DIR, "retriever_k_sweep_timings.csv")

# Experiment settings
QUERY = "List the main topics in this document"
FILE_PATH = "./data/eu.pdf"
RUNS = 3                # measured runs per (vectorstore, k)
WARMUP = True           # perform one warm-up retrieval per (vectorstore, k)
VECTORSTORES = ["faiss", "chroma", "pinecone", "weaviate"]  # change as needed

# k values to sweep 
K_VALUES = [1, 3, 5, 10, 200]

# Load documents once
docs = load_file(FILE_PATH)
if not docs:
    raise ValueError("No documents found!")

# Split documents once
chunks = split_documents(
    splitter_name="recursive",
    documents=docs,
    chunk_size=500,
    chunk_overlap=50,
)

# Load embeddings once
emb_model = load_embeddings_model(
    provider="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def initialize_vectorstore(vs_name):
    kwargs = {}
    # for Pinecone/Weaviate you might need extra config; keep index_name param for pinecone
    if vs_name == "pinecone":
        kwargs["index_name"] = "testindex"
    return build_vectorstore(name=vs_name, chunks=chunks, embeddings_model=emb_model, **kwargs)

def measure_retriever(retriever, query):
    start = time.time()
    docs = retriever.invoke(query)
    end = time.time()
    return round(end - start, 4), len(docs)

def main():
    # Create CSV and header
    with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
        fieldnames = ["vectorstore", "k", "run", "retrieval_time", "docs_retrieved"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over vectorstores
        for vs_name in VECTORSTORES:
            vectorstore = initialize_vectorstore(vs_name)

            # For each k value, create a retriever and measure
            for k in K_VALUES:
                # Build dense retriever configured with k
                retriever = Retriever(retriever_type="dense", vectorstore=vectorstore, k=k)

                # warm-up (once) if requested
                if WARMUP:
                    _ = measure_retriever(retriever, QUERY)

                # measured runs
                for run in range(1, RUNS + 1):
                    retrieval_time, docs_count = measure_retriever(retriever, QUERY)
                    writer.writerow({
                        "vectorstore": vs_name,
                        "k": k,
                        "run": run,
                        "retrieval_time": retrieval_time,
                        "docs_retrieved": docs_count
                    })

if __name__ == "__main__":
    main()
