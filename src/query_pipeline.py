"""
Phase 5 - Full query pipeline orchestration.

Runs every user query through the complete pipeline in order: personal
advice guard (safety.py), medical synonym expansion (synonyms.py), HyDE
query translation, hybrid retrieval with entity boosting and evidence-
weighted reranking (retrieval.py), CRAG confidence check, answer
generation and groundedness verification (generation.py), and mandatory
disclaimer append (safety.py). Ties together every other module in
src/ into a single callable entry point for the app.
"""

from src import safety, retrieval, generation


def answer_query(user_query):
    """Run the full MedSearch Lite pipeline on one user query, Steps A-K."""

    # Step A - personal advice guard
    classification = safety.classify_query(user_query)
    if classification == "PERSONAL":
        return safety.personal_advice_refusal()

    # Steps C-G - HyDE, hybrid retrieval, entity boost, rerank, parent lookup
    hyde_text = retrieval.generate_hyde(user_query)
    semantic_results = retrieval.semantic_search(hyde_text, n_results=20)
    keyword_results = retrieval.bm25_search(user_query, n_results=20)
    merged = retrieval.reciprocal_rank_fusion([semantic_results, keyword_results])
    boosted = retrieval.entity_boost(merged, user_query)
    reranked = retrieval.rerank(boosted, user_query, top_k=5)
    abstract_score_pairs = retrieval.get_parent_abstracts(reranked)

    # Steps H-J - CRAG confidence check, generation, groundedness check
    answer = generation.answer_from_abstracts(abstract_score_pairs, user_query)
    if answer is None:
        return safety.honest_refusal()

    # Step K - mandatory disclaimer
    return safety.append_disclaimer(answer)



def test_pipeline():
    """Run a few different queries through the full end-to-end pipeline."""
    queries = [
        "What is the effect of SGLT2 inhibitors on kidney disease progression?",
        "Should I take metformin for my diabetes?",  # should be caught as PERSONAL
    ]
    for q in queries:
        print("=" * 100)
        print(f"QUERY: {q}\n")
        result = answer_query(q)
        print(result)
        print()


if __name__ == "__main__":
    test_pipeline()

