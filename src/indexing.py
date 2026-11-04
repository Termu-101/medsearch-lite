"""
Phase 4 - Building indexes.

Embeds child chunk propositions with two models (BiomedBERT and
all-MiniLM-L6-v2) for ablation comparison, storing each in its own
Chroma collection under data/indexes/. Also builds a BM25 index over
the original (non-proposition) child chunk text using rank-bm25, and
persists the entity lookup table built in enrichment.py.
"""

# Imports and setup

import os
import json
import pickle
from sentence_transformers import SentenceTransformer, models
import chromadb
from rank_bm25 import BM25Okapi

# MiniLM is a ready-made sentence-transformers model
minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# BiomedBERT needs a manual pooling layer added on top
biomed_transformer = models.Transformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
biomed_pooling = models.Pooling(biomed_transformer.get_word_embedding_dimension())
biomedbert_model = SentenceTransformer(modules=[biomed_transformer, biomed_pooling])


# Load the enriched child chunks

def load_children(filepath):
    """Load the enriched child chunks JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Build a Chroma collection for one embedding model

def build_chroma_collection(children, model, collection_name, persist_dir):
    """Embed every chunk's proposition with the given model and store it in a Chroma collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    propositions = [c["proposition"] for c in children]
    ids = [c["chunk_id"] for c in children]
    metadatas = [
        {
            "parent_id": c["parent_id"],
            "sentence": c["sentence"],
        }
        for c in children
    ]

    print(f"Embedding {len(propositions)} propositions with {collection_name}...")
    embeddings = model.encode(propositions, show_progress_bar=True, batch_size=32).tolist()

    # Chroma has a max batch size per .add() call, so add in smaller batches
    add_batch_size = 1000
    for start in range(0, len(ids), add_batch_size):
        end = start + add_batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            documents=propositions[start:end],
        )

    print(f"Saved {collection_name} to {persist_dir}")


# Build the BM25 index

def build_bm25_index(children, filepath):
    """Build a BM25 index over raw sentences and save it to disk."""
    tokenized_corpus = [c["sentence"].lower().split() for c in children]
    bm25 = BM25Okapi(tokenized_corpus)

    chunk_ids = [c["chunk_id"] for c in children]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
    print(f"Saved BM25 index to {filepath}")


# Small sample test first

def test_sample(n=10):
    """Build tiny indexes on a handful of chunks to sanity-check the pipeline before the full run."""
    children = load_children("data/processed/children.json")[:n]

    build_chroma_collection(children, minilm_model, "test_minilm", "data/indexes/test_chroma")
    build_chroma_collection(children, biomedbert_model, "test_biomedbert", "data/indexes/test_chroma")
    build_bm25_index(children, "data/indexes/test_bm25.pkl")

    # quick sanity query
    client = chromadb.PersistentClient(path="data/indexes/test_chroma")
    collection = client.get_collection("test_minilm")
    query_vector = minilm_model.encode(["metformin and blood sugar control"]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=3)
    print(results)


def main():
    children = load_children("data/processed/children.json")
    print(f"Total chunks to index: {len(children)}")

    build_chroma_collection(children, minilm_model, "minilm_collection", "data/indexes/chroma_minilm")
    build_chroma_collection(children, biomedbert_model, "biomedbert_collection", "data/indexes/chroma_biomedbert")
    build_bm25_index(children, "data/indexes/bm25_index.pkl")

    print("Done.")


if __name__ == "__main__":
    main()
      
