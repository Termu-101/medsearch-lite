"""
Phase 6 - Evaluation.

Runs the query pipeline against the 30-question golden dataset
(golden_dataset.json, sourced from pubmed_qa / medmcqa diabetes
subsets) and computes retrieval metrics (recall@5, MRR), generation
metrics (faithfulness, answer relevance via LLM-as-judge), and
hallucination rate. Repeats this across 6 ablation configurations
(naive -> full pipeline) and saves the comparison table and scores
to evaluation/results/.
"""

import json
import os
import random
from datasets import load_dataset
from groq import RateLimitError
from src import retrieval, generation

random.seed(42)  # reproducible sampling

DIABETES_KEYWORDS = [
    "diabetes", "diabetic", "glucose", "insulin", "hba1c", "glycemic",
    "glycaemic", "metformin", "hyperglycemia", "hypoglycemia", "glycated hemoglobin"
]


def is_diabetes_related(text):
    """Check if a piece of text mentions diabetes-related keywords."""
    text_lower = text.lower()
    return any(k in text_lower for k in DIABETES_KEYWORDS)


def build_from_pubmed_qa(n=3):
    """Pull a small, UNFILTERED sample of diabetes-relevant Q&A pairs from pubmed_qa.

    Deliberately not filtered for corpus relevance - these act as an honest external
    stress test. Expect low recall/MRR on these; that's a real, reportable limitation
    (narrow single-topic corpus vs. broad external questions), not a bug.
    """
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    candidates = []
    for row in dataset:
        question = row["question"]
        long_answer = row["long_answer"]
        if is_diabetes_related(question) or is_diabetes_related(long_answer):
            candidates.append({
                "question": question,
                "reference_answer": long_answer,
                "source_dataset": "pubmed_qa_external",
            })

    print(f"Found {len(candidates)} diabetes-related pubmed_qa questions")
    return random.sample(candidates, min(n, len(candidates)))


def build_from_medmcqa(n=3):
    """Pull a small, UNFILTERED sample of diabetes-relevant Q&A pairs from medmcqa.

    Same rationale as build_from_pubmed_qa: an honest external stress test, not filtered.
    """
    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")

    option_keys = ["opa", "opb", "opc", "opd"]

    candidates = []
    for row in dataset:
        question = row["question"]
        if not is_diabetes_related(question):
            continue
        correct_idx = row["cop"]  # 0-3 index of correct option
        if correct_idx not in [0, 1, 2, 3]:
            continue
        correct_answer = row[option_keys[correct_idx]]
        candidates.append({
            "question": question,
            "reference_answer": correct_answer,
            "source_dataset": "medmcqa_external",
        })

    print(f"Found {len(candidates)} diabetes-related medmcqa questions")
    return random.sample(candidates, min(n, len(candidates)))


def generate_question_from_abstract(parent_abstract):
    """Ask an LLM to write a natural clinical research question + short reference answer
    that the given abstract would answer. Guarantees the question is answerable by our
    own corpus, while still being a paraphrase (not copy-pasted text), so retrieval still
    has real work to do (bridging vocabulary via HyDE, finding the right chunk among 5,779).
    """
    prompt = (
        "Read this medical abstract. Write ONE natural clinical research question that "
        "this abstract directly answers, as if a medical student or junior doctor were "
        "asking it. Then write a SHORT (1-2 sentence) reference answer based only on this "
        "abstract.\n\n"
        "Respond in exactly this format, nothing else:\n"
        "QUESTION: <question>\n"
        "ANSWER: <short answer>\n\n"
        f"Title: {parent_abstract['title']}\n"
        f"Abstract: {parent_abstract['abstract']}"
    )
    response = generation.client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()

    question, answer = "", ""
    for line in text.split("\n"):
        if line.strip().startswith("QUESTION:"):
            question = line.split("QUESTION:", 1)[1].strip()
        elif line.strip().startswith("ANSWER:"):
            answer = line.split("ANSWER:", 1)[1].strip()

    return question, answer


def build_from_own_corpus(n=24):
    """Generate n golden questions directly from our own 499-abstract corpus.

    Guarantees every question is actually answerable by our data (no relevance filtering
    needed), while still testing real retrieval since questions are LLM-paraphrased, not
    copy-pasted from the source abstract.
    """
    parents = json.load(open("data/processed/parents.json", encoding="utf-8"))
    sampled = random.sample(parents, min(n, len(parents)))

    golden = []
    for i, p in enumerate(sampled):
        question, answer = generate_question_from_abstract(p)
        if not question:
            print(f"  [{i+1}/{len(sampled)}] skipped (empty question)")
            continue
        golden.append({
            "question": question,
            "reference_answer": answer,
            "source_dataset": "own_corpus",
            "source_pmid": p["parent_id"],
        })
        print(f"  [{i+1}/{len(sampled)}] {question[:80]}")

    return golden


def build_golden_dataset():
    """Build the 30-question golden evaluation dataset: mostly corpus-derived questions
    (reliable, guaranteed answerable) plus a small external stress-test subset.
    """
    own_corpus_questions = build_from_own_corpus(n=24)
    pubmed_questions = build_from_pubmed_qa(n=3)
    medmcqa_questions = build_from_medmcqa(n=3)

    golden = own_corpus_questions + pubmed_questions + medmcqa_questions
    random.shuffle(golden)

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/golden_dataset.json", "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(golden)} questions to evaluation/golden_dataset.json "
          f"({len(own_corpus_questions)} own_corpus, {len(pubmed_questions)} pubmed_qa_external, "
          f"{len(medmcqa_questions)} medmcqa_external)")


def retrieve_and_judge(question):
    """Run Steps C-G retrieval for one question, then judge each result's relevance."""
    hyde_text = retrieval.generate_hyde(question)
    semantic_results = retrieval.semantic_search(hyde_text, n_results=20)
    keyword_results = retrieval.bm25_search(question, n_results=20)
    merged = retrieval.reciprocal_rank_fusion([semantic_results, keyword_results])
    boosted = retrieval.entity_boost(merged, question)
    reranked = retrieval.rerank(boosted, question, top_k=5)
    abstract_score_pairs = retrieval.get_parent_abstracts(reranked)

    abstracts = [a for a, score, confidence in abstract_score_pairs]
    grades = [generation.grade_abstract(a, question) for a in abstracts]
    relevance = [g != "INCORRECT" for g in grades]  # True = relevant (CORRECT or AMBIGUOUS)

    return relevance


def recall_at_5(relevance_list):
    """1 if at least one of the top 5 results was judged relevant, else 0."""
    return 1 if any(relevance_list) else 0


def reciprocal_rank(relevance_list):
    """1/rank of the first relevant result, or 0 if none were relevant."""
    for i, is_relevant in enumerate(relevance_list):
        if is_relevant:
            return 1 / (i + 1)
    return 0

def evaluate_retrieval(limit=None):
    """Compute Recall@5 and MRR across the golden dataset (or the first `limit` questions)."""
    golden = json.load(open("evaluation/golden_dataset.json", encoding="utf-8"))
    if limit:
        golden = golden[:limit]

    recalls = []
    rrs = []
    for i, item in enumerate(golden):
        question = item["question"]
        try:
            relevance = retrieve_and_judge(question)
        except RateLimitError:
            print(f"Rate limit hit after {i}/{len(golden)} questions. "
                  f"Reporting partial results from what was checked so far.")
            break

        recalls.append(recall_at_5(relevance))
        rrs.append(reciprocal_rank(relevance))
        print(f"[{i+1}/{len(golden)}] Recall@5={recalls[-1]} RR={rrs[-1]:.2f} | {question[:70]}")

    if not recalls:
        print("No questions were evaluated.")
        return {"recall@5": None, "mrr": None}

    mean_recall = sum(recalls) / len(recalls)
    mean_rr = sum(rrs) / len(rrs)

    print(f"\nRecall@5: {mean_recall:.3f}")
    print(f"MRR: {mean_rr:.3f}")

    return {"recall@5": mean_recall, "mrr": mean_rr}


def judge_answer_relevance(question, answer):
    """LLM judges whether a generated answer actually addresses the question asked."""
    prompt = (
        "Does the following answer actually address the question asked, regardless of "
        "whether the answer is fully certain or hedged? Respond with exactly one word: "
        "RELEVANT or IRRELEVANT.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}"
    )
    response = generation.client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().upper() == "RELEVANT"


def evaluate_generation(limit=None):
    """Compute faithfulness, answer relevance, hallucination rate, and refusal rate."""
    golden = json.load(open("evaluation/golden_dataset.json", encoding="utf-8"))
    if limit:
        golden = golden[:limit]

    faithfulness_scores = []
    relevance_results = []
    refusals = 0

    for i, item in enumerate(golden):
        question = item["question"]
        try:
            hyde_text = retrieval.generate_hyde(question)
            semantic_results = retrieval.semantic_search(hyde_text, n_results=20)
            keyword_results = retrieval.bm25_search(question, n_results=20)
            merged = retrieval.reciprocal_rank_fusion([semantic_results, keyword_results])
            boosted = retrieval.entity_boost(merged, question)
            reranked = retrieval.rerank(boosted, question, top_k=5)
            abstract_score_pairs = retrieval.get_parent_abstracts(reranked)

            answer = generation.answer_from_abstracts(abstract_score_pairs, question)

            if answer is None:
                refusals += 1
                print(f"[{i+1}/{len(golden)}] REFUSED | {question[:70]}")
                continue

            abstracts = [a for a, score, conf in abstract_score_pairs]
            score = generation.groundedness_check(answer, abstracts)
            is_relevant = judge_answer_relevance(question, answer)
        except RateLimitError:
            print(f"Rate limit hit after {i}/{len(golden)} questions. Reporting partial results.")
            break

        faithfulness_scores.append(score)
        relevance_results.append(is_relevant)
        print(f"[{i+1}/{len(golden)}] Faithfulness={score:.2f} Relevant={is_relevant} | {question[:60]}")

    n_answered = len(faithfulness_scores)
    mean_faithfulness = sum(faithfulness_scores) / n_answered if n_answered else None
    mean_relevance = sum(relevance_results) / n_answered if n_answered else None
    hallucination_rate = 1 - mean_faithfulness if mean_faithfulness is not None else None
    refusal_rate = refusals / len(golden)

    print(f"\nFaithfulness: {mean_faithfulness}")
    print(f"Answer Relevance: {mean_relevance}")
    print(f"Hallucination Rate: {hallucination_rate}")
    print(f"Refusal Rate: {refusal_rate:.3f} ({refusals}/{len(golden)})")

    return {
        "faithfulness": mean_faithfulness,
        "answer_relevance": mean_relevance,
        "hallucination_rate": hallucination_rate,
        "refusal_rate": refusal_rate,
    }


# Ablation study - 6 progressively more complete retrieval configurations

def get_parent_abstracts_simple(chunk_ids, top_k=5):
    """Dedupe a plain list of chunk_ids down to top_k unique parent abstracts, no scores.
    Used for configs that skip reranking (which normally produces scored tuples).
    """
    seen = set()
    abstracts = []
    for cid in chunk_ids:
        chunk = retrieval.children_lookup.get(cid)
        if not chunk:
            continue
        pid = chunk["parent_id"]
        if pid not in seen:
            seen.add(pid)
            abstracts.append(retrieval.parent_lookup[pid])
        if len(abstracts) >= top_k:
            break
    return abstracts


def retrieve_configurable(query, use_biomedbert=True, use_bm25=False,
                           use_entity_boost=False, use_rerank=False, use_hyde=False, top_k=5):
    """Run retrieval with specific pipeline components enabled/disabled, for ablation comparison.
    Always returns a plain list of parent abstract dicts (no scores), regardless of which
    components were used, so every config can be evaluated the same way.
    """
    query_text_for_embedding = retrieval.generate_hyde(query) if use_hyde else query

    model = retrieval.biomedbert_model if use_biomedbert else retrieval.minilm_model
    collection_path = "data/indexes/chroma_biomedbert" if use_biomedbert else "data/indexes/chroma_minilm"
    collection_name = "biomedbert_collection" if use_biomedbert else "minilm_collection"

    client_chroma = retrieval.chromadb.PersistentClient(path=collection_path)
    collection = client_chroma.get_collection(collection_name)
    query_vector = model.encode([query_text_for_embedding]).tolist()
    dense_ids = collection.query(query_embeddings=query_vector, n_results=20)["ids"][0]

    if use_bm25:
        bm25_ids = retrieval.bm25_search(query, n_results=20)
        merged_pairs = retrieval.reciprocal_rank_fusion([dense_ids, bm25_ids])
    else:
        merged_pairs = [(cid, 1 / (i + 1)) for i, cid in enumerate(dense_ids)]

    if use_entity_boost:
        candidate_ids = retrieval.entity_boost(merged_pairs, query)
    else:
        candidate_ids = [cid for cid, score in merged_pairs]

    if use_rerank:
        final_triples = retrieval.rerank(candidate_ids, query, top_k=top_k)
        abstract_triples = retrieval.get_parent_abstracts(final_triples)
        return [a for a, score, confidence in abstract_triples]
    else:
        return get_parent_abstracts_simple(candidate_ids, top_k=top_k)


ABLATION_CONFIGS = {
    "1_naive_minilm":        dict(use_biomedbert=False, use_bm25=False, use_entity_boost=False, use_rerank=False, use_hyde=False),
    "2_biomedbert_only":     dict(use_biomedbert=True,  use_bm25=False, use_entity_boost=False, use_rerank=False, use_hyde=False),
    "3_plus_bm25":           dict(use_biomedbert=True,  use_bm25=True,  use_entity_boost=False, use_rerank=False, use_hyde=False),
    "4_plus_entity_rerank":  dict(use_biomedbert=True,  use_bm25=True,  use_entity_boost=True,  use_rerank=True,  use_hyde=False),
    "5_plus_hyde":           dict(use_biomedbert=True,  use_bm25=True,  use_entity_boost=True,  use_rerank=True,  use_hyde=True),
}


def evaluate_config(config_params, questions):
    """Compute Recall@5 and MRR for one ablation configuration over a fixed question subset."""
    recalls, rrs = [], []
    for item in questions:
        question = item["question"]
        try:
            abstracts = retrieve_configurable(question, top_k=5, **config_params)
            grades = [generation.grade_abstract(a, question) for a in abstracts]
        except RateLimitError:
            print("    Rate limit hit - stopping this config early.")
            break

        relevance = [g != "INCORRECT" for g in grades]
        recalls.append(1 if any(relevance) else 0)
        rr = 0
        for i, is_relevant in enumerate(relevance):
            if is_relevant:
                rr = 1 / (i + 1)
                break
        rrs.append(rr)

    if not recalls:
        return {"recall@5": None, "mrr": None, "n": 0}
    return {"recall@5": sum(recalls) / len(recalls), "mrr": sum(rrs) / len(rrs), "n": len(recalls)}


def run_ablation_study(subset_size=10):
    """Run all 6 ablation configs on a fixed subset of the golden dataset, save results."""
    golden = json.load(open("evaluation/golden_dataset.json", encoding="utf-8"))
    subset = golden[:subset_size]

    results = {}
    for name, params in ABLATION_CONFIGS.items():
        print(f"\n=== Config: {name} ===")
        results[name] = evaluate_config(params, subset)
        print(f"  {results[name]}")

    # config 6: full pipeline + CRAG/groundedness - measured on generation metrics instead,
    # since CRAG/groundedness affect answer quality, not retrieval ranking
    print("\n=== Config: 6_full_pipeline (generation metrics) ===")
    gen_results = evaluate_generation(limit=subset_size)
    results["6_full_pipeline"] = gen_results

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Ablation Summary ===")
    for name, r in results.items():
        print(f"{name}: {r}")

    return results


if __name__ == "__main__":
    run_ablation_study(subset_size=10)






