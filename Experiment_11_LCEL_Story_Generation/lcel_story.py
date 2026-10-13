import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Load the Gemini key from the project root .env file.
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
load_dotenv(os.path.join(project_root, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the project .env file."
    )


# Call Gemini and return its generated text.
def call_gemini(prompt_value):
    prompt_text = prompt_value.to_string()

    response = requests.post(
        API_URL,
        params={"key": API_KEY},
        json={
            "model": MODEL,
            "input": prompt_text,
            "generation_config": {"temperature": 0},
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    # Extract the latest text from the Interactions API response.
    for step in reversed(result.get("steps", [])):
        for item in reversed(step.get("content", [])):
            if item.get("type") == "text" and item.get("text"):
                return item["text"].strip()

    raise ValueError("No text was found in the Gemini response.")


# Prompt template.
prompt = ChatPromptTemplate.from_template(
    "Write a very short story (2-3 sentences) about {topic}."
)

# Wrap the Gemini API function as an LCEL runnable.
llm = RunnableLambda(call_gemini)

# Parse the output as a string.
parser = StrOutputParser()

# LCEL pipe syntax: prompt -> LLM call -> string parser.
chain = prompt | llm | parser

# Hardcoded topics.
topics = [
    "a dragon and a knight",
    "a lost spaceship",
]


def main():
    results = []

    print("=" * 55)
    print("EXPERIMENT 11: LCEL STORY GENERATION")
    print(f"Model: {MODEL}")
    print("Chain: prompt | Gemini API runnable | parser")
    print("=" * 55)

    for topic in topics:
        print(f"\nTopic: {topic}")

        try:
            story = chain.invoke({"topic": topic})
            print(f"Story: {story}")
            results.append(f"Topic: {topic}\nStory: {story}\n")
        except requests.RequestException as error:
            # Do not print request details that might expose credentials.
            message = (
                f"Gemini API request failed ({type(error).__name__}). "
                "Check API access, quota, and connectivity."
            )
            print(f"Error: {message}")
            results.append(f"Topic: {topic}\nError: {message}\n")
        except (ValueError, RuntimeError) as error:
            message = f"{type(error).__name__}: {error}"
            print(f"Error: {message}")
            results.append(f"Topic: {topic}\nError: {message}\n")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "story_results.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("EXPERIMENT 11: LCEL STORY GENERATION\n")
        file.write(f"Model: {MODEL}\n")
        file.write("Chain: prompt | Gemini API runnable | parser\n\n")
        file.write("\n".join(results))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()