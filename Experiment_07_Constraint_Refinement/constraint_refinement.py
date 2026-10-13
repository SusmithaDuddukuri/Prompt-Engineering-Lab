import os
import re
import requests
from dotenv import load_dotenv

# Load API key from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"
TEMPERATURE = 0.7

INITIAL_PROMPT = """Describe a smartwatch.
Use exactly 3 bullet points.
Each bullet point must start with a verb.
Do not use the word "smart" anywhere.
Keep the entire description under 60 words."""

REFINED_PROMPT = """Describe a fitness watch.
Follow every constraint carefully:
1. Write exactly 3 bullet points and no introductory or concluding text.
2. Each bullet must start with an action verb.
3. Do not use the prohibited word "smart" anywhere in your response.
4. Keep the entire description under 60 words.

Follow this format and use different, relevant details:
- Tracks daily activity.
- Displays messages and alerts.
- Monitors your workout progress.

Return only the three bullet points."""


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
        # Avoid printing request details that might contain the API key.
        return f"ERROR: API request failed (HTTP {status}). Check your API settings."
    except requests.exceptions.RequestException:
        return "ERROR: Network request failed. Check your internet connection."


def check_constraints(text):
    """Check word count, bullet count, verb starts, and prohibited word."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    bullets = [
        re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
        for line in lines
        if re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)", line)
    ]

    # Heuristic list of common action verbs.
    # This is not a complete grammar checker.
    verbs = {
        "track", "tracks",
        "monitor", "monitors",
        "display", "displays",
        "measure", "measures",
        "receive", "receives",
        "customize", "customizes",
        "record", "records",
        "count", "counts",
        "show", "shows",
        "provide", "provides",
        "help", "helps",
        "connect", "connects",
        "deliver", "delivers",
        "last", "lasts",
        "guide", "guides",
        "alert", "alerts",
        "support", "supports",
        "sync", "syncs",
        "offer", "offers",
        "keep", "keeps",
        "check", "checks",
        "analyze", "analyzes",
        "improve", "improves",
        "enable", "enables",
        "display", "displays",
        "notify", "notifies",
        "protect", "protects",
        "calculate", "calculates",
        "provide", "provides",
        "track", "tracks"
    }

    verb_results = []

    for bullet in bullets:
        first_word = re.sub(r"^[^A-Za-z]+", "", bullet).split()
        starts_with_verb = (
            bool(first_word)
            and first_word[0].lower() in verbs
        )
        verb_results.append(starts_with_verb)

    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    prohibited_word_found = bool(
        re.search(r"\bsmart\b", text, re.IGNORECASE)
    )

    non_bullet_lines = [
        line for line in lines
        if not re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)", line)
    ]

    return {
        "bullet_count": len(bullets),
        "exactly_three_bullets": len(bullets) == 3,
        "verb_checks": verb_results,
        "all_bullets_start_with_verbs": (
            len(bullets) == 3 and all(verb_results)
        ),
        "word_count": word_count,
        "under_60_words": word_count <= 60,
        "prohibited_word_found": prohibited_word_found,
        "no_extra_text": len(non_bullet_lines) == 0,
        "bullets": bullets,
    }


def print_report(name, text):
    """Print the response and its constraint-checking report."""
    report = check_constraints(text)

    print(f"\n=== {name} ===")
    print("\nResponse:")
    print(text)

    print("\nConstraint Report:")
    print(
        f"Exactly 3 bullet points: "
        f"{'PASS' if report['exactly_three_bullets'] else 'FAIL'} "
        f"(found {report['bullet_count']})"
    )

    if report["verb_checks"]:
        for index, passed in enumerate(report["verb_checks"], start=1):
            print(
                f"Bullet {index} starts with a recognized verb: "
                f"{'PASS' if passed else 'REVIEW'}"
            )
    else:
        print("Bullet verb checks: FAIL (no bullets detected)")

    print(
        f"Prohibited word absent: "
        f"{'PASS' if not report['prohibited_word_found'] else 'FAIL'}"
    )
    print(
        f"Word count <= 60: "
        f"{'PASS' if report['under_60_words'] else 'FAIL'} "
        f"({report['word_count']} words)"
    )
    print(
        f"No extra text outside bullets: "
        f"{'PASS' if report['no_extra_text'] else 'FAIL'}"
    )

    return report


def main():
    if not API_KEY:
        print(
            "ERROR: GEMINI_API_KEY was not found "
            "in the project root .env file."
        )
        return

    print("=" * 65)
    print("EXPERIMENT 7: CONSTRAINT SPECIFICATION AND REFINEMENT")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 65)

    print("\nInitial prompt:")
    print(INITIAL_PROMPT)
    initial_response = get_response(INITIAL_PROMPT)

    if initial_response.startswith("ERROR:"):
        print(initial_response)
        return

    initial_report = print_report(
        "INITIAL RESPONSE", initial_response
    )

    print("\n" + "=" * 65)
    print("\nRefined prompt:")
    print(REFINED_PROMPT)
    refined_response = get_response(REFINED_PROMPT)

    if refined_response.startswith("ERROR:"):
        print(refined_response)
        return

    refined_report = print_report(
        "REFINED RESPONSE", refined_response
    )

    print("\n" + "=" * 65)
    print("=== COMPARISON SUMMARY ===")

    checks = [
        ("Exactly 3 bullets", "exactly_three_bullets"),
        ("Every bullet starts with a verb",
         "all_bullets_start_with_verbs"),
        ("Prohibited word avoided", None),
        ("At most 60 words", "under_60_words"),
        ("No extra text", "no_extra_text"),
    ]

    for label, key in checks:
        if key is None:
            before = not initial_report["prohibited_word_found"]
            after = not refined_report["prohibited_word_found"]
        else:
            before = initial_report[key]
            after = refined_report[key]

        print(
            f"{label}: Initial={'PASS' if before else 'FAIL/REVIEW'}; "
            f"Refined={'PASS' if after else 'FAIL/REVIEW'}"
        )

    print(
        "\nNote: Verb recognition uses a predefined list. "
        "Review bullet openings manually if the script reports REVIEW."
    )

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "refinement_results.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            "EXPERIMENT 7: CONSTRAINT SPECIFICATION AND REFINEMENT\n"
        )
        file.write(f"Model: {MODEL}\nTemperature: {TEMPERATURE}\n\n")
        file.write("=== INITIAL PROMPT ===\n")
        file.write(INITIAL_PROMPT + "\n\n")
        file.write("=== INITIAL RESPONSE ===\n")
        file.write(initial_response + "\n\n")
        file.write("=== INITIAL CONSTRAINT REPORT ===\n")
        file.write(str(initial_report) + "\n\n")
        file.write("=== REFINED PROMPT ===\n")
        file.write(REFINED_PROMPT + "\n\n")
        file.write("=== REFINED RESPONSE ===\n")
        file.write(refined_response + "\n\n")
        file.write("=== REFINED CONSTRAINT REPORT ===\n")
        file.write(str(refined_report) + "\n\n")
        file.write("=== COMPARISON SUMMARY ===\n")

        for label, key in checks:
            if key is None:
                before = not initial_report["prohibited_word_found"]
                after = not refined_report["prohibited_word_found"]
            else:
                before = initial_report[key]
                after = refined_report[key]

            file.write(
                f"{label}: Initial={'PASS' if before else 'FAIL/REVIEW'}; "
                f"Refined={'PASS' if after else 'FAIL/REVIEW'}\n"
            )

        file.write(
            "\nNote: Verb recognition is heuristic; "
            "review the outputs manually.\n"
        )

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()