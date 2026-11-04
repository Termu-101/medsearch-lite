"""
Phase 5, Steps A & K - Personal advice guard and disclaimer.

Classifies incoming queries as RESEARCH or PERSONAL using an LLM,
issuing a hard refusal for PERSONAL queries before any retrieval
happens. Also defines the mandatory disclaimer text that is appended
in code to every successful answer, independent of any prompt, so it
cannot be bypassed or overridden.
"""

# Set up imports and the LLM client

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Write the query classifier (Step A)

def classify_query(user_query):
    """Classify a user query as RESEARCH or PERSONAL using an LLM."""
    prompt = (
        "Classify the following question as either RESEARCH or PERSONAL.\n"
        "RESEARCH = asking about medical literature, evidence, studies, or general clinical knowledge.\n"
        "PERSONAL = asking for medical advice about the user's own or someone specific's health situation "
        "(e.g. 'should I take X', 'what dose should I use', 'do I have X').\n"
        "Respond with only one word: RESEARCH or PERSONAL.\n\n"
        f"Question: {user_query}"
    )
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().upper()


# Write the refusal message

def personal_advice_refusal():
    """Fixed refusal message for personal medical advice queries."""
    return (
        "I can't provide personal medical advice. I'm a literature research tool designed to help you "
        "explore published evidence on clinical topics. If you have a personal health question, please "
        "consult a qualified healthcare provider. You're welcome to rephrase your question as a general "
        "research question (e.g. 'What does the evidence say about X for Y condition?')."
    )


# Write the honest refusal (for low-confidence retrieval)

def honest_refusal():
    """Fixed refusal message when retrieved evidence isn't confident enough to answer."""
    return (
        "I couldn't find sufficiently reliable evidence in the available literature to answer this "
        "question with confidence. Please try rephrasing your question, or consult a healthcare "
        "provider for guidance."
    )


# Write the mandatory disclaimer

DISCLAIMER = (
    "\n\n---\n"
    "This information is drawn from published medical literature and is intended for research "
    "purposes only. It is not medical advice. Always consult a qualified healthcare provider for "
    "decisions about your health."
)

def append_disclaimer(answer_text):
    """Append the mandatory disclaimer to a generated answer. Cannot be bypassed by prompts."""
    return answer_text + DISCLAIMER


def test_classifier():
    """Sanity-check the RESEARCH/PERSONAL classifier on a few example queries."""
    examples = [
        "What does the evidence say about metformin and cardiovascular outcomes?",
        "Should I take metformin for my diabetes?",
        "What is the effect of SGLT2 inhibitors on kidney disease progression?",
        "I have high blood sugar, what should I do?",
        "I have high bp check the research paper about it and do research.",
    ]
    for q in examples:
        result = classify_query(q)
        print(f"[{result}] {q}")


if __name__ == "__main__":
    test_classifier()

