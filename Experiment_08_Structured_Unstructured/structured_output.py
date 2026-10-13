import os
import requests
from dotenv import load_dotenv

# Load API key from the project root .env file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"
TEMPERATURE = 0.2

UNSTRUCTURED_PROMPT = """
Describe Python, Java, and C++ in one short paragraph.
Mention their typing systems, common use cases, and relative difficulty.
Use plain text, not a table.
"""

STRUCTURED_PROMPT = """
Compare Python, Java, and C++ using a Markdown table.

Use exactly these columns:
Language | Typing (static/dynamic) | Primary Use Case | Difficulty (Beginner/Intermediate/Advanced)

Include exactly three data rows: Python, Java, and C++.
Use the appropriate typing category and a concise primary use case.
Choose one difficulty level for each language.
Return only the Markdown table, with no introduction or conclusion.
"""


def get_response(prompt):
    """Send a prompt to the Gemini Interactions API."""
    payload = {
        "model": MODEL,
        "input": prompt,
        "generation_config": {"temperature": TEMPERATURE}
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

        for step in reversed(result.get("steps", [])):
            for item in reversed(step.get("content", [])):
                if item.get("type") == "text" and item.get("text"):
                    return item["text"].strip()

        return "ERROR: No text response returned."

    except requests.exceptions.Timeout:
        return "ERROR: Request timed out."
    except requests.exceptions.HTTPError as error:
        status = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )
        return (
            f"ERROR: API request failed (HTTP {status}). "
            "Check your API settings."
        )
    except requests.exceptions.RequestException:
        return (
            "ERROR: Network request failed. "
            "Check your internet connection."
        )


def main():
    if not API_KEY:
        print(
            "ERROR: GEMINI_API_KEY was not found "
            "in the project root .env file."
        )
        return

    print("=" * 75)
    print("EXPERIMENT 8: STRUCTURED VS. UNSTRUCTURED OUTPUT")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 75)

    print("\n=== UNSTRUCTURED PROMPT ===")
    print(UNSTRUCTURED_PROMPT.strip())

    unstructured = get_response(UNSTRUCTURED_PROMPT)

    if unstructured.startswith("ERROR:"):
        print(unstructured)
        return

    print("\n=== UNSTRUCTURED RESPONSE ===")
    print(unstructured)

    print("\n" + "=" * 75)

    print("\n=== STRUCTURED PROMPT ===")
    print(STRUCTURED_PROMPT.strip())

    structured = get_response(STRUCTURED_PROMPT)

    if structured.startswith("ERROR:"):
        print(structured)
        return

    print("\n=== STRUCTURED RESPONSE (MARKDOWN TABLE) ===")
    print(structured)

    # Basic checks for the requested table content
    print("\n=== STRUCTURED OUTPUT CHECKS ===")

    required_columns = [
        "Language",
        "Typing",
        "Primary Use Case",
        "Difficulty"
    ]

    for column in required_columns:
        found = column.lower() in structured.lower()
        print(
            f"Column '{column}' present: "
            f"{'PASS' if found else 'REVIEW'}"
        )

    for language in ["Python", "Java", "C++"]:
        found = language.lower() in structured.lower()
        print(
            f"Language '{language}' present: "
            f"{'PASS' if found else 'REVIEW'}"
        )

    # Save both responses and a short comparison
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_results.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("EXPERIMENT 8: STRUCTURED VS. UNSTRUCTURED OUTPUT\n")
        file.write(f"Model: {MODEL}\nTemperature: {TEMPERATURE}\n\n")

        file.write("=== UNSTRUCTURED PROMPT ===\n")
        file.write(UNSTRUCTURED_PROMPT.strip() + "\n\n")
        file.write("=== UNSTRUCTURED RESPONSE ===\n")
        file.write(unstructured + "\n\n")

        file.write("=== STRUCTURED PROMPT ===\n")
        file.write(STRUCTURED_PROMPT.strip() + "\n\n")
        file.write("=== STRUCTURED RESPONSE ===\n")
        file.write(structured + "\n\n")

        file.write("=== ANALYSIS ===\n")
        file.write(
            "The unstructured paragraph presents the information "
            "as natural prose and is useful for a quick general overview.\n\n"
        )
        file.write(
            "The Markdown table places each language in a separate row "
            "and each characteristic in a separate column. This makes "
            "the information easier to scan and compare consistently.\n\n"
        )
        file.write(
            "The basic checks verify that the expected column labels "
            "and language names appear. They do not guarantee that every "
            "cell is factually correct or that the Markdown table is "
            "perfectly formatted, so inspect the table manually.\n"
        )

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()