
import os
import re
import requests
from dotenv import load_dotenv

# Load the API key from the project root .env file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"
TEMPERATURE = 0.7

# Same core question in all three prompts
QUESTION = "Explain the importance of exercise."

PROMPTS = {
    "BASELINE": QUESTION,

    "ROLE-BASED (Fitness Coach)": (
        "You are a motivational fitness coach.\n"
        "Explain the importance of exercise."
    ),

    "ROLE + NEGATIVE": (
        "You are a motivational fitness coach.\n"
        "Do not use any medical jargon.\n"
        "Do not mention weight loss.\n\n"
        "Explain the importance of exercise."
    ),
}


def get_response(prompt):
    """Send one prompt to the Gemini Interactions API."""
    payload = {
        "model": MODEL,
        "input": prompt,
        "generation_config": {
            "temperature": TEMPERATURE
        }
    }

    try:
        response = requests.post(
            API_URL,
            params={"key": API_KEY},
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        # Extract the latest text response from the API result
        for step in reversed(result.get("steps", [])):
            for item in reversed(step.get("content", [])):
                if item.get("type") == "text" and item.get("text"):
                    return item["text"].strip()

        return "ERROR: No text response was returned."

    except requests.exceptions.Timeout:
        return "ERROR: The request timed out."
    except requests.exceptions.HTTPError as error:
        # Avoid printing the request URL, which may contain the API key
        status = error.response.status_code if error.response is not None else "unknown"
        details = (
            error.response.text[:500]
            if error.response is not None
            else str(error)
        )
        return f"ERROR: API request failed (HTTP {status}). Details: {details}"
    except requests.exceptions.RequestException as error:
        return f"ERROR: Network request failed: {error}"


def analyse_constraints(text):
    """Check common indicators; these checks cannot detect all medical jargon."""
    lower_text = text.lower()

    weight_loss_terms = [
        "weight loss",
        "lose weight",
        "losing weight",
        "burn fat",
        "fat loss",
        "shed pounds",
    ]

    jargon_terms = [
        "cardiovascular",
        "metabolism",
        "hypertension",
        "glucose",
        "aerobic capacity",
        "musculoskeletal",
        "endocrine",
        "insulin sensitivity",
    ]

    weight_matches = [
        term for term in weight_loss_terms if term in lower_text
    ]
    jargon_matches = [
        term for term in jargon_terms if term in lower_text
    ]

    return weight_matches, jargon_matches


def main():
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY was not found in the project root .env file.")
        return

    results = {}

    print("=" * 65)
    print("EXPERIMENT 6: ROLE-BASED AND NEGATIVE PROMPTING")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 65)

    # Send all three prompts using the same model and temperature
    for name, prompt in PROMPTS.items():
        print(f"\n=== {name} ===")
        print(f"Prompt:\n{prompt}\n")

        answer = get_response(prompt)
        results[name] = answer
        print(f"Response:\n{answer}\n")
        print("-" * 65)

    # Compare tone, vocabulary and adherence
    baseline = results["BASELINE"]
    role_based = results["ROLE-BASED (Fitness Coach)"]
    role_negative = results["ROLE + NEGATIVE"]

    weight_matches, jargon_matches = analyse_constraints(role_negative)

    print("\n=== ANALYSIS ===")

    print("\n1. TONE")
    print(
        "Compare the baseline with the fitness-coach responses. "
        "Look for encouragement, direct address, enthusiasm, and motivation."
    )

    print("\n2. VOCABULARY")
    print(
        "Compare the wording in all three responses. The role-based prompts "
        "may use more conversational and motivational language."
    )

    print("\n3. NEGATIVE CONSTRAINT ADHERENCE")
    if weight_matches:
        print("Weight-loss constraint: POSSIBLE VIOLATION.")
        print("Matching phrases:", ", ".join(weight_matches))
    else:
        print("Weight-loss constraint: No listed weight-loss phrases detected.")

    if jargon_matches:
        print("Medical-jargon check: Review these terms manually:")
        print(", ".join(jargon_matches))
    else:
        print("Medical-jargon check: No listed jargon terms detected.")

    print(
        "Note: These keyword checks are only a first-pass screening. "
        "Read the full response to decide whether it mentions weight loss "
        "or uses other medical jargon."
    )

    # Save prompts, responses, and analysis
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_results.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("EXPERIMENT 6: ROLE-BASED AND NEGATIVE PROMPTING\n")
        file.write(f"Model: {MODEL}\nTemperature: {TEMPERATURE}\n\n")

        for name, prompt in PROMPTS.items():
            file.write(f"=== {name} ===\n")
            file.write(f"Prompt:\n{prompt}\n\n")
            file.write(f"Response:\n{results[name]}\n\n")

        file.write("=== ANALYSIS ===\n")
        file.write(
            "Tone: Compare the baseline and fitness-coach responses "
            "for encouragement and motivational language.\n"
        )
        file.write(
            "Vocabulary: Compare conversational wording and technical terms.\n"
        )
        file.write(
            "Weight-loss phrase matches: "
            + (", ".join(weight_matches) if weight_matches else "None detected")
            + "\n"
        )
        file.write(
            "Medical-jargon phrase matches: "
            + (", ".join(jargon_matches) if jargon_matches else "None detected")
            + "\n"
        )
        file.write(
            "Note: Keyword checks are not exhaustive; review the responses manually.\n"
        )

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()
