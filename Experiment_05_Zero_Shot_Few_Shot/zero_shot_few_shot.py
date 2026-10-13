import os
import re
import requests
from dotenv import load_dotenv

# Load API key from the project root .env file
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
load_dotenv(os.path.join(project_root, ".env"))

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check the project root .env file."
    )

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"

# Five fixed test reviews and their expected sentiments
test_reviews = [
    {
        "review": "The battery life is amazing!",
        "actual": "Positive"
    },
    {
        "review": "It works, nothing special.",
        "actual": "Neutral"
    },
    {
        "review": "The screen stopped working after two days.",
        "actual": "Negative"
    },
    {
        "review": "The package arrived yesterday.",
        "actual": "Neutral"
    },
    {
        "review": "Excellent camera and beautiful photos.",
        "actual": "Positive"
    }
]

# Three fixed examples for few-shot prompting
examples = [
    ('"Best phone ever!"', "Positive"),
    ('"The camera is slow."', "Negative"),
    ('"Average quality, nothing special."', "Neutral")
]


def ask_llm(prompt):
    """Send a prompt to Gemini and return its text response."""
    try:
        response = requests.post(
            API_URL,
            params={"key": api_key},
            json={
                "model": MODEL,
                "input": prompt
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        for step in reversed(result.get("steps", [])):
            for item in step.get("content", []):
                if item.get("type") == "text" and item.get("text"):
                    return item["text"].strip()

        return "ERROR: No text response received."

    except requests.exceptions.Timeout:
        return "ERROR: Request timed out."

    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Gemini API."

    except requests.exceptions.HTTPError as e:
        return f"ERROR: API request failed ({e})."

    except (KeyError, TypeError, ValueError) as e:
        return f"ERROR: Could not parse response ({e})."


def extract_label(response):
    """Extract one valid sentiment label from a response."""
    if response.startswith("ERROR:"):
        return "ERROR"

    # Accept a response only if it contains one clear label.
    matches = re.findall(
        r"\b(Positive|Neutral|Negative)\b",
        response,
        flags=re.IGNORECASE
    )

    if len(matches) == 1:
        return matches[0].capitalize()

    return "INVALID"


def build_zero_shot_prompt(review):
    return (
        "Classify the customer review as Positive, Neutral, or Negative. "
        "Return exactly one label and no explanation.\n\n"
        f'Review: "{review}"'
    )


def build_few_shot_prompt(review):
    example_text = "\n".join(
        f"Review: {example_review} -> {label}"
        for example_review, label in examples
    )

    return (
        "Classify each customer review as Positive, Neutral, or Negative. "
        "Return exactly one label and no explanation.\n\n"
        f"{example_text}\n\n"
        f'Review: "{review}" ->'
    )


results = []

print("=" * 75)
print("EXPERIMENT 5: ZERO-SHOT VS FEW-SHOT SENTIMENT CLASSIFICATION")
print("=" * 75)
print("Model:", MODEL)
print("Test reviews:", len(test_reviews))
print("Few-shot examples:", len(examples))

for index, item in enumerate(test_reviews, start=1):
    review = item["review"]
    actual = item["actual"]

    print("\n" + "-" * 75)
    print(f"REVIEW {index}: {review}")
    print("Ground truth:", actual)

    zero_response = ask_llm(build_zero_shot_prompt(review))
    zero_label = extract_label(zero_response)

    few_response = ask_llm(build_few_shot_prompt(review))
    few_label = extract_label(few_response)

    print("\nZero-shot response:", zero_response)
    print("Zero-shot label:", zero_label)

    print("\nFew-shot response:", few_response)
    print("Few-shot label:", few_label)

    results.append({
        "review": review,
        "actual": actual,
        "zero_response": zero_response,
        "zero_label": zero_label,
        "few_response": few_response,
        "few_label": few_label
    })

# Calculate accuracy using the same ground truth for both methods
zero_correct = sum(
    row["zero_label"] == row["actual"] for row in results
)
few_correct = sum(
    row["few_label"] == row["actual"] for row in results
)

total = len(results)
zero_accuracy = zero_correct / total * 100
few_accuracy = few_correct / total * 100

# Format adherence: percentage returning exactly one valid label
zero_format = sum(
    row["zero_label"] in ("Positive", "Neutral", "Negative")
    for row in results
) / total * 100

few_format = sum(
    row["few_label"] in ("Positive", "Neutral", "Negative")
    for row in results
) / total * 100

# Agreement between zero-shot and few-shot predictions
agreement = sum(
    row["zero_label"] == row["few_label"] for row in results
) / total * 100

print("\n" + "=" * 75)
print("COMPARISON TABLE")
print("=" * 75)
print(
    f"{'Review':<8} {'Actual':<10} {'Zero-shot':<12} "
    f"{'Few-shot':<12} {'Agree?':<8}"
)
print("-" * 60)

for index, row in enumerate(results, start=1):
    agree = "Yes" if row["zero_label"] == row["few_label"] else "No"
    print(
        f"{index:<8} {row['actual']:<10} "
        f"{row['zero_label']:<12} {row['few_label']:<12} {agree:<8}"
    )

print("\n" + "=" * 75)
print("EVALUATION SUMMARY")
print("=" * 75)
print(f"Zero-shot accuracy: {zero_correct}/{total} ({zero_accuracy:.1f}%)")
print(f"Few-shot accuracy:  {few_correct}/{total} ({few_accuracy:.1f}%)")
print(f"Zero-shot format adherence: {zero_format:.1f}%")
print(f"Few-shot format adherence:  {few_format:.1f}%")
print(f"Prediction agreement: {agreement:.1f}%")

print("\nAnalysis:")
if few_accuracy > zero_accuracy:
    print("Few-shot prompting achieved higher accuracy on this test set.")
elif few_accuracy < zero_accuracy:
    print("Zero-shot prompting achieved higher accuracy on this test set.")
else:
    print("Both methods achieved the same accuracy on this test set.")

print(
    "Format adherence measures whether each response contained exactly "
    "one recognized sentiment label. Prediction agreement measures how "
    "often both methods chose the same label."
)

# Save all responses and the comparison to a text file
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "comparison_results.txt")

with open(output_file, "w", encoding="utf-8") as file:
    file.write("EXPERIMENT 5: ZERO-SHOT VS FEW-SHOT\n\n")

    for index, row in enumerate(results, start=1):
        file.write(f"Review {index}: {row['review']}\n")
        file.write(f"Ground truth: {row['actual']}\n")
        file.write(f"Zero-shot response: {row['zero_response']}\n")
        file.write(f"Zero-shot label: {row['zero_label']}\n")
        file.write(f"Few-shot response: {row['few_response']}\n")
        file.write(f"Few-shot label: {row['few_label']}\n\n")

    file.write("COMPARISON TABLE\n")
    file.write(
        f"Zero-shot accuracy: {zero_correct}/{total} "
        f"({zero_accuracy:.1f}%)\n"
    )
    file.write(
        f"Few-shot accuracy: {few_correct}/{total} "
        f"({few_accuracy:.1f}%)\n"
    )
    file.write(f"Zero-shot format adherence: {zero_format:.1f}%\n")
    file.write(f"Few-shot format adherence: {few_format:.1f}%\n")
    file.write(f"Prediction agreement: {agreement:.1f}%\n")

print("\nFull results saved to:")
print(output_file)