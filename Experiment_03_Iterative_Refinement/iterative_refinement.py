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
        "GEMINI_API_KEY not found. Check the project root .env file."
    )

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"


def ask_llm(prompt):
    """Send a prompt to Gemini and return the text response."""
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


# Iteration 0: Baseline prompt
prompt0 = "Explain photosynthesis to a 10-year-old."

# Iteration 1: Add an analogy, brevity, and simple language
prompt1 = (
    "You are a friendly science teacher. Explain photosynthesis "
    "to a 10-year-old using a simple kitchen analogy. Use exactly "
    "3 short sentences and avoid technical jargon."
)

# Iteration 2: Keep the first refinement and add memory and a diagram
prompt2 = (
    "You are a friendly science teacher. Explain photosynthesis "
    "to a 10-year-old using a simple kitchen analogy. Use exactly "
    "3 short sentences and avoid technical jargon. Then add a "
    "memory trick and describe a simple text-only diagram."
)

prompts = [prompt0, prompt1, prompt2]
responses = []

# Save all prompts and responses to a file beside this script
output_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    "responses.txt"
)

with open(output_file, "w", encoding="utf-8") as file:
    file.write("EXPERIMENT 3: ITERATIVE PROMPT REFINEMENT\n")

    for i, prompt in enumerate(prompts):
        print("\n" + "=" * 60)

        if i == 0:
            heading = "ITERATION 0 (BASELINE)"
        elif i == 1:
            heading = "ITERATION 1 (ANALOGY + BREVITY)"
        else:
            heading = "ITERATION 2 (MEMORY TRICK + DIAGRAM)"

        print(heading)
        print("=" * 60)
        print("\nPrompt:")
        print(prompt)

        answer = ask_llm(prompt)
        responses.append(answer)

        print("\nResponse:")
        print(answer)

        if i == 0:
            analysis = (
                "The baseline establishes a starting point. "
                "Check whether the explanation is simple and concise."
            )
        elif i == 1:
            analysis = (
                "The analogy, simple language, and sentence limit "
                "aim to make the explanation easier for a child."
            )
        else:
            analysis = (
                "The memory trick and diagram description add "
                "ways to remember and visualize the concept."
            )

        print("\nAnalysis:")
        print(analysis)

        file.write(f"\n{'=' * 60}\n{heading}\n{'=' * 60}\n")
        file.write(f"\nPrompt:\n{prompt}\n")
        file.write(f"\nResponse:\n{answer}\n")
        file.write(f"\nAnalysis:\n{analysis}\n")

    summary = (
        "\nFinal Summary:\n"
        "Iteration 0: Provides the baseline explanation.\n"
        "Iteration 1: Adds a kitchen analogy, brevity, and simple language.\n"
        "Iteration 2: Adds a memory trick and a text-only diagram description.\n"
        "Compare the responses to see whether each refinement improves "
        "clarity, memorability, and visualization.\n"
    )

    print("\n" + "=" * 60)
    print(summary)
    file.write("\n" + summary)

print("All prompts and responses saved to:")
print(output_file)