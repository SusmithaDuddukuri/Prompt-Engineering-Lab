import os
import re
import requests
from dotenv import load_dotenv

# Locate the project root and load the private API key.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"
TEMPERATURE = 0

PROBLEM = (
    "A farmer has 15 cows. He buys 7 more cows. "
    "Then he sells half of his cows. How many cows does he have left?"
)
EXPECTED_ANSWER = 11


def ask_gemini(prompt):
    """Send a prompt to Gemini and return its text response."""
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Check the project's .env file."
        )

    response = requests.post(
        API_URL,
        params={"key": API_KEY},
        json={
            "model": MODEL,
            "input": prompt,
            "generation_config": {"temperature": TEMPERATURE},
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    # Find the most recent text content in the response.
    for step in reversed(result.get("steps", [])):
        for item in reversed(step.get("content", [])):
            if item.get("type") == "text" and item.get("text"):
                return item["text"].strip()

    raise ValueError("No text answer was found in the API response.")


def extract_final_number(answer):
    """Extract the final numeric answer from a response."""
    matches = re.findall(r"-?\d+", answer.replace(",", ""))
    return int(matches[-1]) if matches else None


def main():
    direct_prompt = PROBLEM
    cot_prompt = PROBLEM + "\nLet's think step by step. Show each step."

    print("=" * 60)
    print("EXPERIMENT 10: CHAIN-OF-THOUGHT PROMPTING")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 60)

    print("\n=== DIRECT PROMPT ===")
    print(f"Problem: {PROBLEM}")
    direct_response = ask_gemini(direct_prompt)
    direct_answer = extract_final_number(direct_response)
    print(f"Response: {direct_response}")
    print(f"Extracted answer: {direct_answer}")

    print("\n=== CHAIN-OF-THOUGHT PROMPT ===")
    print(f"Prompt: {cot_prompt}")
    cot_response = ask_gemini(cot_prompt)
    cot_answer = extract_final_number(cot_response)
    print(f"Response: {cot_response}")
    print(f"Extracted answer: {cot_answer}")

    print("\n=== ANALYSIS ===")
    print(f"Expected answer: {EXPECTED_ANSWER}")
    print(
        "Direct prompt: "
        + ("PASS" if direct_answer == EXPECTED_ANSWER else "FAIL")
    )
    print(
        "Chain-of-Thought prompt: "
        + ("PASS" if cot_answer == EXPECTED_ANSWER else "FAIL")
    )

    if direct_answer == EXPECTED_ANSWER and cot_answer == EXPECTED_ANSWER:
        conclusion = (
            "Both prompts produced the correct answer. "
            "The step-by-step response provides a clearer explanation, "
            "but this single example does not prove that CoT is always more accurate."
        )
    elif cot_answer == EXPECTED_ANSWER:
        conclusion = (
            "The Chain-of-Thought prompt produced the correct answer, "
            "while the direct prompt did not, in this example."
        )
    elif direct_answer == EXPECTED_ANSWER:
        conclusion = (
            "The direct prompt was correct, while the Chain-of-Thought prompt "
            "was not correct in this example."
        )
    else:
        conclusion = (
            "Neither extracted answer matched the expected answer. "
            "Inspect the responses and extraction results."
        )

    print(f"Conclusion: {conclusion}")

    # Save both responses and the comparison for the lab record.
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_results.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("EXPERIMENT 10: CHAIN-OF-THOUGHT PROMPTING\n")
        file.write(f"Model: {MODEL}\nTemperature: {TEMPERATURE}\n\n")
        file.write("=== DIRECT PROMPT ===\n")
        file.write(f"Prompt: {direct_prompt}\n")
        file.write(f"Response: {direct_response}\n")
        file.write(f"Extracted answer: {direct_answer}\n\n")
        file.write("=== CHAIN-OF-THOUGHT PROMPT ===\n")
        file.write(f"Prompt: {cot_prompt}\n")
        file.write(f"Response: {cot_response}\n")
        file.write(f"Extracted answer: {cot_answer}\n\n")
        file.write("=== ANALYSIS ===\n")
        file.write(f"Expected answer: {EXPECTED_ANSWER}\n")
        file.write(
            f"Direct prompt: "
            f"{'PASS' if direct_answer == EXPECTED_ANSWER else 'FAIL'}\n"
        )
        file.write(
            f"Chain-of-Thought prompt: "
            f"{'PASS' if cot_answer == EXPECTED_ANSWER else 'FAIL'}\n"
        )
        file.write(f"Conclusion: {conclusion}\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as error:
        print(f"API request failed: {error}")
        print("Check your internet connection and Gemini API configuration.")
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")