import os
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
        "GEMINI_API_KEY not found. Check the .env file."
    )

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"


def send_prompt(prompt):
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
                    return item["text"]

        return "Error: No text response received."

    except requests.exceptions.Timeout:
        return "API/Network Error: Request timed out."

    except requests.exceptions.ConnectionError:
        return "API/Network Error: Could not connect to Gemini."

    except requests.exceptions.HTTPError as e:
        return f"API Error: {e}\n{response.text}"

    except (KeyError, TypeError, ValueError) as e:
        return f"Response Error: {e}"


# Baseline prompt: general instruction
baseline_prompt = "Explain what recursion is."

# Enhanced prompt: role, audience, analogy, and format
enhanced_prompt = (
    "You are a computer science professor teaching beginners. "
    "Explain recursion using a simple real-world analogy. "
    "Write exactly two paragraphs using clear, simple language."
)

# Send both prompts separately using the same model
baseline_response = send_prompt(baseline_prompt)
enhanced_response = send_prompt(enhanced_prompt)

print("=" * 60)
print("EXPERIMENT 2: PROMPT ENGINEERING COMPARISON")
print("=" * 60)

print("\nBASELINE PROMPT")
print("-" * 60)
print(baseline_prompt)
print("\nBASELINE RESPONSE")
print(baseline_response)

print("\nENHANCED PROMPT")
print("-" * 60)
print(enhanced_prompt)
print("\nENHANCED RESPONSE")
print(enhanced_response)

print("\nCOMPARISON / ANALYSIS")
print("-" * 60)
print("Baseline response words:", len(baseline_response.split()))
print("Enhanced response words:", len(enhanced_response.split()))
print(
    "\nThe baseline prompt asks a general question. The enhanced "
    "prompt specifies the role, audience, use of an analogy, and "
    "two-paragraph format. These instructions guide the model toward "
    "a more structured and beginner-friendly explanation."
)