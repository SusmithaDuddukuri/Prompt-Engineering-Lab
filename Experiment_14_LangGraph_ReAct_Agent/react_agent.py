import ast
import operator
import os

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "agent_results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-lite"


# --------------------------------------------------
# TOOL 1: CURRENT WEATHER
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a city using Open-Meteo."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=15,
        )
        geo_response.raise_for_status()

        results = geo_response.json().get("results", [])
        if not results:
            return f"City not found: {city}"

        place = results[0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m",
                "temperature_unit": "celsius",
            },
            timeout=15,
        )
        weather_response.raise_for_status()

        current = weather_response.json().get("current", {})
        temperature = current.get("temperature_2m")

        if temperature is None:
            return f"Temperature unavailable for {city}."

        return (
            f"Current temperature in {place.get('name', city)}: "
            f"{temperature} degrees Celsius (Open-Meteo)."
        )

    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"Weather request failed: {exc}"


# --------------------------------------------------
# TOOL 2: SAFE CALCULATOR
# --------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_calculate(expression):
    """Evaluate basic arithmetic without executing arbitrary Python code."""
    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")

            result = _ALLOWED_OPERATORS[type(node.op)](left, right)

            if abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd)
        ):
            return -evaluate(node.operand) if isinstance(
                node.op, ast.USub
            ) else evaluate(node.operand)

        raise ValueError("Only basic arithmetic is allowed.")

    return evaluate(tree)


@tool
def calculate(expression: str) -> str:
    """Calculate arithmetic expressions using numbers and +, -, *, /."""
    try:
        return str(safe_calculate(expression))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"


# --------------------------------------------------
# MAIN LANGGRAPH AGENT
# --------------------------------------------------

def main():
    trace = []

    def log(message):
        print(message, flush=True)
        trace.append(str(message))

    log("=" * 60)
    log("EXPERIMENT 14: LANGGRAPH REACT AGENT")
    log("=" * 60)
    log(f"Model: {MODEL}")
    log("Tools: get_weather, calculate")

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the project .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=API_KEY,
        temperature=0,
        max_retries=1,
        timeout=30,
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather, calculate],
        system_prompt=(
            "You are a helpful assistant with weather and calculator tools. "
            "For the user's request, call get_weather for London. "
            "Use calculate to convert Celsius to Fahrenheit using "
            "(Celsius * 9 / 5) + 32. Use calculate to find 15% of 320, "
            "then use calculate to add that result to the Fahrenheit value. "
            "Base the answer on actual tool results. If a tool fails, "
            "explain the failure instead of inventing a value."
        ),
    )

    query = (
        "What is the current temperature in London in Fahrenheit? "
        "Calculate 15% of 320 and add it to the Fahrenheit temperature. "
        "Use the tools to perform the calculations."
    )

    log("\n=== AGENT EXECUTION TRACE ===")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 10},
    )

    for message in result["messages"]:
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for call in tool_calls:
                log(f"\nAction: {call['name']}")
                log(f"Action Input: {call['args']}")

        elif getattr(message, "type", "") == "tool":
            content = message.content
            if isinstance(content, str):
                log(f"Observation ({message.name}): {content}")
            else:
                log(f"Observation ({message.name}): {content}")

    last_message = result["messages"][-1]
    content = last_message.content

    if isinstance(content, str):
        final_answer = content
    elif isinstance(content, list):
        final_answer = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )
    else:
        final_answer = str(content)

    log("\n=== FINAL ANSWER ===")
    log(final_answer)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(trace))
        file.write("\n")

    log(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()