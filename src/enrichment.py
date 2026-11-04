"""
Phase 3 - SciSpacy entity enrichment.

Runs the en_core_sci_lg model over every child chunk to extract
CHEMICAL (drug) and DISEASE (condition) entities, stores them as
chunk metadata, and builds an entity_name -> [chunk_ids] lookup
table used at query time to boost retrieval for chunks matching
query entities.
"""

# Imports and setup

import os
import json
import spacy

nlp_ner = spacy.load("en_ner_bc5cdr_md")

# Load the child chunks

def load_children(filepath):
    """Load the child chunks JSON saved by chunking.py."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Extract entities from one chunk's sentence

def extract_entities(text):
    """Run NER on a sentence and return separate lists of chemical and disease entity strings."""
    doc = nlp_ner(text)
    chemicals = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
    diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    return chemicals, diseases


# Enrich all chunks with entities

def enrich_chunks(children):
    """Add chemical and disease entity lists to every child chunk, with progress logging."""
    for i, chunk in enumerate(children):
        chemicals, diseases = extract_entities(chunk["sentence"])
        chunk["chemicals"] = chemicals
        chunk["diseases"] = diseases
        if (i + 1) % 500 == 0:
            print(f"Enriched {i + 1}/{len(children)} chunks...")
    return children


# Build the entity → chunk_ids lookup table

def build_entity_lookup(children):
    """Build a reverse lookup: entity name -> list of chunk_ids mentioning it."""
    lookup = {}
    for chunk in children:
        for entity in chunk["chemicals"] + chunk["diseases"]:
            key = entity.lower().strip()
            lookup.setdefault(key, []).append(chunk["chunk_id"])
    return lookup


# Save results

def save_json(data, filepath):
    """Save any JSON-serializable object to disk, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {filepath}")


# Small sample test first

def test_sample(n=5):
    """Run entity extraction on a few chunks only, to sanity-check output."""
    children = load_children("data/processed/children.json")[:n]
    for c in children:
        chemicals, diseases = extract_entities(c["sentence"])
        print(f"\nSENT: {c['sentence']}")
        print(f"CHEMICALS: {chemicals}")
        print(f"DISEASES: {diseases}")


# Full pipeline

def main():
    children = load_children("data/processed/children.json")
    print(f"Total chunks to enrich: {len(children)}")

    enriched = enrich_chunks(children)
    save_json(enriched, "data/processed/children.json")

    lookup = build_entity_lookup(enriched)
    save_json(lookup, "data/processed/entity_lookup.json")

    print("Done.")


if __name__ == "__main__":
    main()



