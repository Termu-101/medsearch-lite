"""
Phase 5, Steps I-J - Answer generation and groundedness checking.

Generates the final cited answer from retrieved abstracts using a
strong LLM (Claude Sonnet or Llama 3.1 70B on Groq), requiring inline
PMID citations, evidence level labels, and uncertainty language.
Then uses a cheap/fast LLM (Claude Haiku or Llama 3.1 8B on Groq) to
extract factual claims from the answer and verify each against the
retrieved abstracts, triggering regeneration (up to 2 attempts) or an
honest refusal if the groundedness score falls below threshold.
"""


# Grading/Checking our generated retrieval abstracts 

import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def grade_abstract(abstract, query):
    """Grade one retrieved abstract as CORRECT, AMBIGUOUS, or INCORRECT relative to the query."""
    prompt = (
        "You are evaluating whether a retrieved abstract is useful evidence for answering a question.\n"
        "Respond with exactly one word: CORRECT, AMBIGUOUS, or INCORRECT.\n"
        "CORRECT = the abstract directly and clearly helps answer the question.\n"
        "AMBIGUOUS = the abstract is topically related but doesn't clearly answer the question.\n"
        "INCORRECT = the abstract is not relevant to the question.\n\n"
        f"Question: {query}\n\n"
        f"Abstract: {abstract['title']} {abstract['abstract']}"
    )
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip().upper()

    # Some models ignore the "exactly one word" instruction and add reasoning text.
    # Parse robustly: find all verdict words mentioned and take the LAST one (models
    # tend to state their final conclusion after reasoning through it). Use word
    # boundaries since "INCORRECT" is a substring-superset of "CORRECT".
    matches = list(re.finditer(r"\bINCORRECT\b|\bAMBIGUOUS\b|\bCORRECT\b", text))
    if not matches:
        return "AMBIGUOUS"  # fallback if no recognizable label was found at all
    return matches[-1].group()


def crag_confidence_check(abstract_score_pairs, query):
    """Grade each abstract and combine with reranker confidence to decide overall confidence.

    abstract_score_pairs: list of (abstract_dict, weighted_score, confidence) tuples,
    from retrieval.get_parent_abstracts(). confidence is a genuine 0-1 sigmoid-scaled value.
    """
    abstracts = [a for a, score, confidence in abstract_score_pairs]
    confidences = [confidence for a, score, confidence in abstract_score_pairs]

    grades = [grade_abstract(a, query) for a in abstracts]
    print("Grades:", grades)
    print("Confidences:", confidences)

    # LOW confidence only if NOTHING usable was found at all - a mix of a few good
    # sources plus some weaker filler is normal and expected, not a failure. Requiring
    # a majority to be CORRECT was too conservative and caused excessive refusals even
    # when a single strong, well-matched source existed.
    has_any_good_grade = any(g != "INCORRECT" for g in grades)
    has_any_good_score = any(c >= 0.35 for c in confidences)

    # only keep the genuinely good abstracts for generation, so the LLM isn't
    # distracted by irrelevant filler that merely survived retrieval's ranking
    good_pairs = [pair for pair, g in zip(abstract_score_pairs, grades) if g != "INCORRECT"]

    if not has_any_good_grade and not has_any_good_score:
        return "LOW", abstract_score_pairs
    if not good_pairs:
        good_pairs = abstract_score_pairs  # fallback: scores alone said HIGH, keep everything
    return "HIGH", good_pairs


# Test CRAG using real retrieval output

def test_crag():
    """Run Steps C-H end to end: real retrieval, then the CRAG confidence check on the result."""
    from src import retrieval

    query = "What is the effect of SGLT2 inhibitors on kidney disease progression?"

    hyde_text = retrieval.generate_hyde(query)
    semantic_results = retrieval.semantic_search(hyde_text, n_results=20)
    keyword_results = retrieval.bm25_search(query, n_results=20)
    merged = retrieval.reciprocal_rank_fusion([semantic_results, keyword_results])
    boosted = retrieval.entity_boost(merged, query)
    reranked = retrieval.rerank(boosted, query, top_k=5)
    abstract_score_pairs = retrieval.get_parent_abstracts(reranked)

    confidence, filtered_pairs = crag_confidence_check(abstract_score_pairs, query)
    print(f"\nFinal confidence: {confidence}, kept {len(filtered_pairs)}/{len(abstract_score_pairs)} abstracts")


# Answer Generation

def generate_answer(abstract_score_pairs, query):
    """Generate a cited answer from retrieved abstracts, using a strong LLM."""
    abstracts = [a for a, score, confidence in abstract_score_pairs]

    sources_text = ""
    for a in abstracts:
        sources_text += (
            f"\n---\nPMID: {a['parent_id']}\n"
            f"Evidence Level: {a['evidence_level']}\n"
            f"Title: {a['title']}\n"
            f"Abstract: {a['abstract']}\n"
        )

    prompt = (
        "You are a clinical literature research assistant. Answer the question using ONLY "
        "the abstracts provided below. Do not use any outside knowledge.\n\n"
        "Requirements:\n"
        "- Every factual claim must have an inline citation in the format (PMID: xxxxx, Level X)\n"
        "- Use uncertainty language throughout (e.g. 'evidence suggests', 'may be associated with') "
        "rather than definitive statements\n"
        "- If the abstracts don't fully answer the question, say so explicitly\n"
        "- Do not give personal medical advice or recommendations\n\n"
        f"Question: {query}\n\n"
        f"Abstracts:{sources_text}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# Groundedness check

# Extract individual factual claims from the generated answer

import spacy

nlp_sentences = spacy.load("en_core_sci_lg", disable=["ner", "tagger", "lemmatizer"])


def extract_claims(answer_text):
    """Split a generated answer into individual sentences using scispacy's deterministic
    sentence splitter, rather than relying on an LLM to reliably follow a one-claim-per-line
    format. Strips markdown bullet markers so sentence boundaries aren't thrown off by them.
    """
    cleaned = answer_text.replace("* ", "").replace("---", "")
    doc = nlp_sentences(cleaned)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# Verify one claim against the source abstracts

def verify_claim(claim, abstracts):
    """Check whether a single claim is actually supported by the retrieved abstracts."""
    sources_text = "\n\n".join(f"{a['title']} {a['abstract']}" for a in abstracts)

    prompt = (
        "Does the following claim accurately reflect information actually present in the "
        "provided source text? Respond with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"Claim: {claim}\n\n"
        f"Source text:\n{sources_text}"
    )
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip().upper()

    # same robustness fix as grade_abstract: take the last verdict word mentioned.
    # Use word boundaries since "SUPPORTED" is a substring of "UNSUPPORTED".
    matches = list(re.finditer(r"\bUNSUPPORTED\b|\bSUPPORTED\b", text))
    if not matches:
        return False  # fallback: treat unparseable response as unsupported (fail safe)
    return matches[-1].group() == "SUPPORTED"


# Compute the overall groundedness score

def groundedness_check(answer_text, abstracts):
    """Extract claims from an answer and check each against the abstracts, returning a score."""
    claims = extract_claims(answer_text)
    if not claims:
        return 1.0  # nothing to verify

    supported = [verify_claim(c, abstracts) for c in claims]
    score = sum(supported) / len(supported)

    print(f"Claims checked: {len(claims)}, supported: {sum(supported)}, score: {score:.2f}")
    return score


# The regeneration loop — wire it into a new top-level function

def generate_with_groundedness_check(abstract_score_pairs, query, max_attempts=2):
    """Generate an answer, verify its groundedness, and regenerate (stricter) if it fails."""
    abstracts = [a for a, score, confidence in abstract_score_pairs]

    for attempt in range(1, max_attempts + 1):
        answer = generate_answer(abstract_score_pairs, query)
        score = groundedness_check(answer, abstracts)

        if score >= 0.80:
            return answer

        print(f"Attempt {attempt} failed groundedness check (score {score:.2f}), retrying...")

    return None  # signals the caller to issue an honest refusal


# Main entry point for query_pipeline.py: runs Steps H-J together

def answer_from_abstracts(abstract_score_pairs, query):
    """Run Steps H-J: CRAG confidence check, generation, and groundedness verification.

    Returns the final answer text if successful, or None if the system should refuse
    (either due to low retrieval confidence, or failing the groundedness check twice).
    """
    confidence, filtered_pairs = crag_confidence_check(abstract_score_pairs, query)
    if confidence == "LOW":
        return None

    return generate_with_groundedness_check(filtered_pairs, query)


def test_generation():
    """Run Steps C-J end to end: retrieval, CRAG check, generation, groundedness check."""
    from src import retrieval

    query = "What is the effect of SGLT2 inhibitors on kidney disease progression?"

    hyde_text = retrieval.generate_hyde(query)
    semantic_results = retrieval.semantic_search(hyde_text, n_results=20)
    keyword_results = retrieval.bm25_search(query, n_results=20)
    merged = retrieval.reciprocal_rank_fusion([semantic_results, keyword_results])
    boosted = retrieval.entity_boost(merged, query)
    reranked = retrieval.rerank(boosted, query, top_k=5)
    abstract_score_pairs = retrieval.get_parent_abstracts(reranked)

    answer = answer_from_abstracts(abstract_score_pairs, query)
    if answer is None:
        print("Refused: low confidence or failed groundedness check twice.")
    else:
        print("ANSWER:\n")
        print(answer)


if __name__ == "__main__":
    test_generation()



