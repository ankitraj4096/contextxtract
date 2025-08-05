# logic_evaluator.py
# Uses OpenRouter API with OpenAI SDK for insurance logic decision

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost"}  # required by OpenRouter
)

def gpt_decision_engine(query, relevant_snippets: list) -> dict:
    # Token-efficient: only send most relevant, short snippets
    top_snips = [s[:400] for s in relevant_snippets[:2]]
    joined = "\n\n".join([f"Clause {i+1}:\n{part}" for i, part in enumerate(top_snips)])
    prompt = f"""
You are an expert insurance policy analyst.

Given these clauses from the policy document:
{joined}

And the user's question:
"{query}"

Answer the following:
1. What is the direct answer? ("yes", "no", or a detailed answer)
2. Your confidence (0-1).
3. Your reasoning (why?).
4. List clauses that support your answer.
5. List any conditions or limitations.
6. If you can't answer, say what extra info is needed.

Return STRICTLY this JSON:
{{
  "answer": "",
  "confidence": 0.0,
  "reasoning": "",
  "supporting_clauses": [],
  "conditions": [],
  "additional_info_needed": []
}}
ONLY output JSON. Do NOT add explanations.
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # supported on OpenRouter, change if you want
            messages=[
                {"role": "system", "content": "Strictly output ONLY the required JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0
        )
        output = response.choices[0].message.content
        decision = json.loads(output)
    except Exception as e:
        print("OpenRouter call failed:", e)
        decision = fallback_decision(query, relevant_snippets)
    return decision

def fallback_decision(query, clauses):
    for c in clauses:
        if "not covered" in c.lower():
            return {
                "answer": "No, not covered.",
                "confidence": 0.6,
                "reasoning": "Found: not covered.",
                "supporting_clauses": [c],
                "conditions": [],
                "additional_info_needed": []
            }
    return {
        "answer": "Yes, covered.",
        "confidence": 0.5,
        "reasoning": "Did not find 'not covered' in chunks.",
        "supporting_clauses": clauses,
        "conditions": [],
        "additional_info_needed": ["Need confirmation from full document."]
    }

if __name__ == "__main__":
    question = "Does this policy cover knee surgery, and what are the conditions?"
    snippets = [
        "Knee surgery is covered under section 4.2.",
        "A 3-month waiting period applies to all major surgeries."
    ]
    print(gpt_decision_engine(question, snippets))
