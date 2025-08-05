# test_llm.py
# Tests llm_parser and logic_evaluator using OpenRouter API

from llm_parser import parse_query_with_gpt
from logic_evaluator import gpt_decision_engine
import json

print("=== Testing Query Parser (llm_parser.py) ===")
test_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
parsed = parse_query_with_gpt(test_query)
print("Parsed Query:")
print(json.dumps(parsed, indent=2, ensure_ascii=False))

mock_relevant_snippets = [
    "Section 4.2: Knee surgery is covered under the policy for insured members after a 3-month waiting period.",
    "General exclusion: Any claim within the first 3 months except in case of accident is not admissible."
]

print("\n=== Testing Decision Engine (logic_evaluator.py) ===")
decision = gpt_decision_engine(test_query, mock_relevant_snippets)
print("Decision Output:")
print(json.dumps(decision, indent=2, ensure_ascii=False))
