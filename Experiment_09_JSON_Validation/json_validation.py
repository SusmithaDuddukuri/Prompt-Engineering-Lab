import os
import json
import requests
from dotenv import load_dotenv

# Load API key from the project root .env file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"
TEMPERATURE = 0


INITIAL_PROMPT = """
Generate a JSON array describing exactly three books.

Each array element must be an object with exactly these keys:
- "title": a string
- "author": a string
- "year": an integer

Example of the required structure:
[
  {
    "title": "Example Book",
    "author": "Example Author",
    "year": 2000
  }
]

Choose three real books and use accurate information.
Output only valid JSON. Do not include Markdown fences or explanations.
"""

STRICT_PROMPT = """
Generate a JSON array describing exactly three real books.

Requirements:
1. Return a JSON array containing exactly three objects.
2. Every object must have exactly these keys: "title", "author", "year".
3. "title" and "author" must be strings.
4. "year" must be an integer.
5. Use double quotes for JSON strings and keys.
6. Do not use trailing commas.
7. Do not include Markdown fences, comments, or explanatory text.
8. Ensure the result can be parsed directly by Python's json.loads().

Return only the JSON array.
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


def validate_books(data):
    """Validate the JSON structure and required field types."""
    if not isinstance(data, list):
        return False, "The JSON root must be an array."

    if len(data) != 3:
        return False, f"Expected exactly 3 books; found {len(data)}."

    required_keys = {"title", "author", "year"}

    for index, book in enumerate(data, start=1):
        if not isinstance(book, dict):
            return False, f"Book {index} is not a JSON object."

        if set(book.keys()) != required_keys:
            return False, (
                f"Book {index} must contain exactly these keys: "
                "title, author, year."
            )

        if not isinstance(book["title"], str):
            return False, f"Book {index}: title must be a string."

        if not isinstance(book["author"], str):
            return False, f"Book {index}: author must be a string."

        # bool is a subclass of int in Python, so exclude it explicitly.
        if not isinstance(book["year"], int) or isinstance(book["year"], bool):
            return False, f"Book {index}: year must be an integer."

    return True, "Valid structure: exactly 3 books with the required keys and types."


def main():
    if not API_KEY:
        print(
            "ERROR: GEMINI_API_KEY was not found "
            "in the project root .env file."
        )
        return

    print("=" * 70)
    print("EXPERIMENT 9: JSON GENERATION AND VALIDATION")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 70)

    prompts = [
        ("INITIAL PROMPT", INITIAL_PROMPT),
        ("STRICT RETRY PROMPT", STRICT_PROMPT)
    ]

    final_data = None
    final_json = None
    output_lines = [
        "EXPERIMENT 9: JSON GENERATION AND VALIDATION",
        f"Model: {MODEL}",
        f"Temperature: {TEMPERATURE}",
        ""
    ]

    for attempt, (name, prompt) in enumerate(prompts, start=1):
        print(f"\n=== {name} ===")
        response_text = get_response(prompt)

        if response_text.startswith("ERROR:"):
            print(response_text)
            output_lines.append(response_text)
            break

        print("\nLLM Response:")
        print(response_text)

        output_lines.append(f"=== {name} ===")
        output_lines.append("LLM Response:")
        output_lines.append(response_text)

        try:
            parsed_data = json.loads(response_text)
            print("\nJSON parsing: PASS")

            valid, message = validate_books(parsed_data)
            print(f"Schema validation: {'PASS' if valid else 'FAIL'}")
            print(message)

            output_lines.append("JSON parsing: PASS")
            output_lines.append(
                f"Schema validation: {'PASS' if valid else 'FAIL'}"
            )
            output_lines.append(message)

            if valid:
                final_data = parsed_data
                final_json = json.dumps(
                    parsed_data, indent=2, ensure_ascii=False
                )
                break

            print("The JSON structure is incorrect; trying the strict prompt.")
            output_lines.append(
                "The JSON structure is incorrect; trying the strict prompt."
            )

        except json.JSONDecodeError as error:
            print(f"\nJSON parsing: FAIL")
            print(f"Error: {error}")
            output_lines.append("JSON parsing: FAIL")
            output_lines.append(f"Error: {error}")

            if attempt == 1:
                print("Retrying with a stricter prompt...")
                output_lines.append("Retrying with a stricter prompt...")

    print("\n" + "=" * 70)
    print("=== FINAL RESULT ===")

    if final_data is not None:
        print("Valid JSON generated successfully.")
        print(final_json)
        output_lines.append("\n=== FINAL RESULT ===")
        output_lines.append("Valid JSON generated successfully.")
        output_lines.append(final_json)
    else:
        print("Could not obtain valid JSON after the retry.")
        output_lines.append("\nCould not obtain valid JSON after the retry.")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "validation_results.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines) + "\n")

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()