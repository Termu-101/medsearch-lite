"""
Phase 5, Steps D-G - Hybrid retrieval, entity boosting, and reranking.

Combines dense semantic search (BiomedBERT Chroma collection, queried
with a HyDE-generated embedding) with keyword search (BM25 on the
synonym-expanded query) via Reciprocal Rank Fusion. Boosts scores for
chunks whose entities match entities extracted from the query, then
reranks the result with a cross-encoder, weighted by each chunk's
evidence level. Finally looks up and deduplicates the parent abstracts
for the top reranked child chunks.
"""

# Imports and setup

import os
import json
import pickle
import math
import chromadb
from sentence_transformers import SentenceTransformer, models
from groq import Groq
from dotenv import load_dotenv
from src.synonyms import get_expansion_terms



load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

biomed_transformer = models.Transformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
biomed_pooling = models.Pooling(biomed_transformer.get_word_embedding_dimension())
biomedbert_model = SentenceTransformer(modules=[biomed_transformer, biomed_pooling])


# Generate the HyDE hypothetical abstract

def generate_hyde(user_query):
    """Generate a hypothetical PubMed abstract excerpt answering the query, for semantic search."""
    prompt = (
        "Write a short, realistic excerpt (2-3 sentences) from a PubMed abstract that would "
        "plausibly answer the following clinical research question. Write in formal academic "
        "medical style, as if reporting study findings. Do not mention that this is hypothetical.\n\n"
        f"Question: {user_query}"
    )
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# Small test harness

def test_hyde():
    """Sanity-check HyDE generation on a couple of example queries."""
    examples = [
        "Does metformin reduce cardiovascular risk in type 2 diabetes patients?",
        "What is the effect of SGLT2 inhibitors on kidney disease progression?",
    ]
    for q in examples:
        hyde_text = generate_hyde(q)
        print(f"QUERY: {q}")
        print(f"HYDE:  {hyde_text}\n")


# BM25

# Load the BM25 index and children data (needed for BM25 search)

def load_bm25_index(filepath="data/indexes/bm25_index.pkl"):
    """Load the pickled BM25 index and its corresponding chunk_ids list."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


bm25_index, bm25_chunk_ids = load_bm25_index()


# Semantic search against the BiomedBERT Chroma collection

def semantic_search(hyde_text, n_results=20):
    """Embed the HyDE text with BiomedBERT and query the Chroma collection for top matches."""
    client_chroma = chromadb.PersistentClient(path="data/indexes/chroma_biomedbert")
    collection = client_chroma.get_collection("biomedbert_collection")

    query_vector = biomedbert_model.encode([hyde_text]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=n_results)

    return results["ids"][0]  # ranked list of chunk_ids


# BM25 keyword search

def bm25_search(user_query, n_results=20, expansion_weight=0.4):
    """Run BM25 search, scoring original query terms and synonym-expansion terms separately.

    The original query terms are weighted more heavily than expansion terms, so specific
    words (e.g. a drug name) aren't drowned out by a long list of expansion synonyms.
    """
    original_tokens = user_query.lower().split()
    expansion_terms = get_expansion_terms(user_query)

    original_scores = bm25_index.get_scores(original_tokens)

    if expansion_terms:
        expansion_tokens = " ".join(expansion_terms).split()
        expansion_scores = bm25_index.get_scores(expansion_tokens)
    else:
        expansion_scores = [0] * len(original_scores)

    combined_scores = [
        o + expansion_weight * e
        for o, e in zip(original_scores, expansion_scores)
    ]

    ranked = sorted(zip(bm25_chunk_ids, combined_scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:n_results]]


# Reciprocal Rank Fusion to merge both ranked lists

def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked lists of chunk_ids using Reciprocal Rank Fusion.

    Returns a list of (chunk_id, score) tuples, ranked highest score first.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Test function for the full pipeline across synonym and safety.py

def test_hybrid_retrieval():
    """Test the full Step D pipeline: HyDE -> semantic search + BM25 -> RRF merge."""
    query = "Does metformin reduce cardiovascular risk in type 2 diabetes patients?"

    hyde_text = generate_hyde(query)
    print(f"HYDE: {hyde_text}\n")

    semantic_results = semantic_search(hyde_text, n_results=20)
    print(f"Semantic search top 5: {semantic_results[:5]}\n")

    keyword_results = bm25_search(query, n_results=20)
    print(f"BM25 top 5: {keyword_results[:5]}\n")

    merged = reciprocal_rank_fusion([semantic_results, keyword_results])
    print(f"Merged (RRF) top 5: {[cid for cid, score in merged[:5]]}")


# Multi-query internal test, checking retrieved text quality across several different topics

def test_multiple_queries():
    """Run several different queries through Steps C-E and print the resulting top-5 sentences."""
    children = json.load(open("data/processed/children.json", encoding="utf-8"))
    lookup = {c["chunk_id"]: c for c in children}

    queries = [
        "Does metformin reduce cardiovascular risk in type 2 diabetes patients?",
        "What is the effect of SGLT2 inhibitors on kidney disease progression?",
        "How does insulin therapy affect glycemic control in type 1 diabetes?",
        "What are the risk factors for diabetic retinopathy?",
        "Is bariatric surgery effective for weight loss in obese diabetic patients?",
    ]

    for query in queries:
        print("=" * 100)
        print(f"QUERY: {query}\n")

        hyde_text = generate_hyde(query)
        semantic_results = semantic_search(hyde_text, n_results=20)
        keyword_results = bm25_search(query, n_results=20)
        merged = reciprocal_rank_fusion([semantic_results, keyword_results])
        boosted = entity_boost(merged, query)

        for cid in boosted[:5]:
            c = lookup.get(cid)
            if c:
                print(f"  - {c['sentence'][:130]}")
        print()


# Entity Boosting

# import and load the entity model

import spacy

nlp_ner = spacy.load("en_ner_bc5cdr_md")

with open("data/processed/entity_lookup.json", "r", encoding="utf-8") as f:
    entity_lookup = json.load(f)

# Extract entities from the query

def extract_query_entities(query):
    """Run NER on the user query and return a set of lowercased entity strings."""
    doc = nlp_ner(query)
    return {ent.text.lower().strip() for ent in doc.ents}

# Boost chunks whose entities match

# entities tagged on this many chunks or more are considered too generic to be a useful signal
COMMON_ENTITY_THRESHOLD = 150


def entity_boost(ranked_chunks, query):
    """Boost chunks whose tagged entities match RARE/specific entities extracted from the query.

    ranked_chunks: list of (chunk_id, score) tuples, e.g. from reciprocal_rank_fusion().
    Common entities (e.g. "diabetes" in a diabetes-focused corpus) are excluded, since matching
    them carries little discriminating signal compared to a specific drug or condition name.
    """
    query_entities = extract_query_entities(query)
    specific_entities = {
        e for e in query_entities
        if e in entity_lookup and len(entity_lookup[e]) < COMMON_ENTITY_THRESHOLD
    }
    if not specific_entities:
        return [chunk_id for chunk_id, score in ranked_chunks]

    boosted_scores = {}
    for chunk_id, score in ranked_chunks:
        matched_entities = [e for e in specific_entities if chunk_id in entity_lookup.get(e, [])]
        if len(matched_entities) >= 2:
            score *= 3.0
        elif len(matched_entities) == 1:
            score *= 2.0

        boosted_scores[chunk_id] = score

    reordered = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in reordered]


# Step F - Cross-encoder reranking, weighted by evidence level

from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

children_data = json.load(open("data/processed/children.json", encoding="utf-8"))
children_lookup = {c["chunk_id"]: c for c in children_data}

parents_data = json.load(open("data/processed/parents.json", encoding="utf-8"))
parent_lookup = {p["parent_id"]: p for p in parents_data}


def rerank(ranked_chunk_ids, query, top_k=5, entity_bonus_weight=1.15):
    """Rerank candidates with a cross-encoder, weighted by evidence level and specific entity match.

    A small entity_bonus_weight is applied on top of the cross-encoder score for chunks matching a
    specific (rare) query entity, so the Step E signal survives rather than being fully overwritten
    by the cross-encoder's from-scratch relevance judgment.
    """
    evidence_weights = {1: 1.3, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8}

    query_entities = extract_query_entities(query)
    specific_entities = {
        e for e in query_entities
        if e in entity_lookup and len(entity_lookup[e]) < COMMON_ENTITY_THRESHOLD
    }

    pairs = []
    valid_ids = []
    for chunk_id in ranked_chunk_ids:
        chunk = children_lookup.get(chunk_id)
        if chunk:
            pairs.append((query, chunk["sentence"]))
            valid_ids.append(chunk_id)

    cross_scores = reranker_model.predict(pairs)

    weighted_scores = []
    for chunk_id, raw_score in zip(valid_ids, cross_scores):
        chunk = children_lookup[chunk_id]

        # raw cross-encoder logits are unbounded, so squash to a genuine 0-1
        # confidence via sigmoid BEFORE applying ranking-purpose weighting
        confidence = 1 / (1 + math.exp(-raw_score))

        score = raw_score
        matched_entities = [e for e in specific_entities if chunk_id in entity_lookup.get(e, [])]
        if matched_entities:
            score *= entity_bonus_weight
        evidence_level = parent_lookup.get(chunk["parent_id"], {}).get("evidence_level", 5)
        weight = evidence_weights.get(evidence_level, 1.0)
        weighted_scores.append((chunk_id, score * weight, confidence))

    weighted_scores.sort(key=lambda x: x[1], reverse=True)
    return weighted_scores[:top_k]  # list of (chunk_id, weighted_score, confidence) tuples


# Step G - Parent abstract lookup and dedupe

def get_parent_abstracts(top_chunks):
    """Look up and deduplicate parent abstracts for the given top (chunk_id, score, confidence) tuples.

    Returns a list of (abstract, score, confidence) tuples, keeping the highest weighted score
    seen for each parent if multiple of its child chunks appear in top_chunks.
    """
    seen_parents = {}
    for chunk_id, score, confidence in top_chunks:
        chunk = children_lookup.get(chunk_id)
        if not chunk:
            continue
        parent_id = chunk["parent_id"]
        if parent_id not in seen_parents or score > seen_parents[parent_id][1]:
            seen_parents[parent_id] = (parent_lookup[parent_id], score, confidence)

    return list(seen_parents.values())


# Full end-to-end test: Steps C through G on one query

def test_full_retrieval():
    """Run the complete Steps C-G pipeline on one query and print the final abstracts."""
    query = "What is the effect of SGLT2 inhibitors on kidney disease progression?"

    hyde_text = generate_hyde(query)
    semantic_results = semantic_search(hyde_text, n_results=20)
    keyword_results = bm25_search(query, n_results=20)
    merged = reciprocal_rank_fusion([semantic_results, keyword_results])
    boosted = entity_boost(merged, query)
    reranked = rerank(boosted, query, top_k=5)
    abstract_score_pairs = get_parent_abstracts(reranked)

    print(f"QUERY: {query}\n")
    print(f"Final unique abstracts returned: {len(abstract_score_pairs)}\n")
    for a, score, confidence in abstract_score_pairs:
        print(f"PMID: {a['parent_id']} | Evidence Level: {a['evidence_level']} | Score: {score:.3f} | Confidence: {confidence:.3f}")
        print(f"Title: {a['title']}")
        print(f"Abstract: {a['abstract'][:200]}...")
        print()


if __name__ == "__main__":
    test_full_retrieval()





