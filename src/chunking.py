"""
Phase 2 - Parent-child chunking.

Splits each raw abstract into a parent chunk (the full abstract, kept
for rich context at generation time) and child chunks (individual
sentences, kept for precise retrieval). Stores the parent_id on every
child chunk for lookup at query time, and generates an LLM proposition
(a clean, one-sentence factual rewrite) for each child chunk so that
embeddings are built on the proposition rather than the raw sentence.
Saves processed chunks to data/processed/.
"""

# Import and Setup

import os
import json
import uuid
import spacy
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
nlp = spacy.load("en_core_sci_lg", disable=["ner", "tagger", "lemmatizer"])


# Load Data

def load_raw_abstracts(filepath):
    """Load the raw PubMed abstracts JSON saved by data_collection.py."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Parent chunk

def make_parent_chunk(record):
    """Wrap one raw abstract record as a parent chunk, keyed by its PMID."""
    return {
        "parent_id": record["pmid"],
        "title": record["title"],
        "abstract": record["abstract"],
        "authors": record["authors"],
        "journal": record["journal"],
        "publication_year": record["publication_year"],
        "publication_types": record["publication_types"],
        "mesh_terms": record["mesh_terms"],
        "evidence_level": record["evidence_level"],
    }


# Child chunks using scispacy's sentence splitter

def make_child_chunks(record):
    """Split one abstract's text into sentence-level child chunks using scispacy."""
    doc = nlp(record["abstract"])
    children = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        children.append({
            "chunk_id": str(uuid.uuid4()),
            "parent_id": record["pmid"],
            "sentence": text,
            "proposition": None,
        })
    return children


# Proposition generation

def generate_proposition(sentence):
    """Call Groq's gpt-oss-20b to rewrite one sentence into a clean, standalone factual statement."""
    prompt = (
        "Rewrite the following sentence from a medical abstract as a single, "
        "clear, standalone factual statement. Do not add information not present "
        "in the sentence. Return only the rewritten sentence, nothing else.\n\n"
        f"Sentence: {sentence}"
    )
    response = client.chat.completions.create(
        model="allam-2-7b",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def main():
    records = load_raw_abstracts("data/raw/pubmed_diabetes_500.json")

    parents = [make_parent_chunk(r) for r in records]
    save_to_json_simple(parents, "data/processed/parents.json")

    checkpoint_path = "data/processed/children.json"

    # resume from existing checkpoint if present, instead of reprocessing everything
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            all_children = json.load(f)
        print(f"Resuming from checkpoint: {len(all_children)} chunks loaded.")
    else:
        all_children = []
        for r in records:
            all_children.extend(make_child_chunks(r))

    remaining = [c for c in all_children if not c["proposition"]]
    print(f"Total chunks: {len(all_children)}, remaining to process: {len(remaining)}")

    batch_size = 50
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        try:
            for chunk in batch:
                chunk["proposition"] = generate_proposition(chunk["sentence"])
        except RateLimitError:
            save_to_json_simple(all_children, checkpoint_path)
            print("Rate limit hit — checkpoint saved. Rerun later once the quota resets.")
            return

        # save progress after every batch
        save_to_json_simple(all_children, checkpoint_path)
        print(f"Processed {start + len(batch)}/{len(remaining)} remaining chunks, saved checkpoint.")

    print("Done.")

def save_to_json_simple(data, filepath):
    """Save any list of dicts as JSON, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Small-sample test harness

def test_sample(n=3):
    """Run chunking + propositions on a few abstracts only, to sanity-check output."""
    records = load_raw_abstracts("data/raw/pubmed_diabetes_500.json")[:n]
    for r in records:
        children = make_child_chunks(r)
        print(f"\n=== PMID {r['pmid']} ({len(children)} sentences) ===")
        for c in children[:3]:  # just first 3 sentences per abstract
            prop = generate_proposition(c["sentence"])
            print(f"RAW:  {c['sentence']}")
            print(f"PROP: {prop}\n")


if __name__ == "__main__":
    main()


