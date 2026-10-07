import os
import requests
from dotenv import load_dotenv

try:
    # Load environment variables
    load_dotenv()

    # Read API key from .env
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables."
        )

    print("Environment ready. Sending prompt...")

    prompt = "What is prompt engineering? Answer in one sentence."

    print("\nPrompt:")
    print(prompt)

    # Gemini Interactions API
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    data = {
        "model": "gemini-flash-lite-latest",
        "input": prompt
    }

    response = requests.post(
        url,
        params={"key": api_key},
        json=data,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    # Extract model response
    answer = result["steps"][-1]["content"][0]["text"]

    print("\nResponse:")
    print(answer)

except ValueError as e:
    print(f"Error: {e}")

except requests.exceptions.Timeout:
    print("API/Network Error: Request timed out.")

except requests.exceptions.ConnectionError:
    print("API/Network Error: Could not connect to Gemini.")

except requests.exceptions.HTTPError as e:
    print(f"API Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")