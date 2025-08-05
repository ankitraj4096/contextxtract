# llm_parser.py
# Uses OpenRouter API with OpenAI SDK to parse insurance questions

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost"}
)

def parse_query_with_gpt(query: str) -> dict:
    prompt = f"""
Parse the following insurance-related question and give only a valid, strict JSON like:
{{
  "original_query": "...",
  "intent": "coverage_check | claim_eligibility | policy_terms | info | ...",
  "entities": {{
    "age": "...",
    "gender": "...",
    "procedure": "...",
    "location": "...",
    "policy_duration": "...",
    "amount": "...",
    "condition": ""
  }},
  "keywords": ["...", "..."]
}}
Query: "{query}"
If an entity doesn't appear, use an empty string. Only output JSON, nothing else.
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're an expert at structuring insurance queries. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0
        )
        output = response.choices[0].message.content
        parsed = json.loads(output)
    except Exception as e:
        print("OpenRouter call failed:", e)
        parsed = {}
    return parsed

if __name__ == "__main__":
    q = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    print(parse_query_with_gpt(q))
