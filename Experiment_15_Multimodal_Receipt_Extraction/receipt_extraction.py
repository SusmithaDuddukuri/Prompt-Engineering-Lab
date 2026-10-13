import base64
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = EXPERIMENT_DIR / "images" / "receipt.png"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "receipt_data.json"
TEXT_OUTPUT_FILE = OUTPUT_DIR / "extraction_results.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-lite"


# --------------------------------------------------
# MULTIMODAL PROMPT
# --------------------------------------------------

RECEIPT_PROMPT = """
You are an accurate receipt information extraction assistant.

Analyze the attached restaurant receipt image and extract:

1. Restaurant name
2. Receipt date in YYYY-MM-DD format, if readable
3. Items ordered as a list containing each item name and its price
4. Subtotal
5. Tax
6. Total

Rules:
- Use only information visible in the image.
- Do not invent missing or unreadable values.
- Use null for missing restaurant name, date, subtotal, tax, or total.
- For unreadable item names or prices, clearly mark the uncertainty.
- Represent monetary amounts as JSON numbers, not currency strings.
- Do not include currency symbols in numeric values.
- Return exactly one valid JSON object with this schema:

{
  "restaurant_name": "string or null",
  "date": "YYYY-MM-DD or null",
  "items": [
    {
      "name": "string",
      "price": 0.00
    }
  ],
  "subtotal": 0.00,
  "tax": 0.00,
  "total": 0.00
}

Do not use Markdown fences or add explanations outside the JSON object.
"""


# --------------------------------------------------
# IMAGE ENCODING
# --------------------------------------------------

def encode_image(image_path):
    """Read an image and return its Base64 representation and MIME type."""
    suffix = image_path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    if suffix not in mime_types:
        raise ValueError("Use a JPG, PNG, or WEBP image.")

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_image, mime_types[suffix]


# --------------------------------------------------
# JSON EXTRACTION AND VALIDATION
# --------------------------------------------------

def parse_json_response(response_text):
    """Parse JSON returned by the vision model."""
    cleaned = response_text.strip()

    # Handle a response wrapped in Markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(cleaned[start:end + 1])


def validate_receipt(data):
    """Check required fields and validate subtotal + tax against total."""
    required_fields = [
        "restaurant_name",
        "date",
        "items",
        "subtotal",
        "tax",
        "total",
    ]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            f"Missing required JSON fields: {', '.join(missing_fields)}"
        )

    validation = {
        "subtotal_plus_tax_matches_total": None,
        "difference": None,
        "status": "NOT_CHECKED",
    }

    subtotal = data.get("subtotal")
    tax = data.get("tax")
    total = data.get("total")

    if subtotal is None or tax is None or total is None:
        validation["status"] = "INCOMPLETE_VALUES"
        return validation

    try:
        subtotal_decimal = Decimal(str(subtotal))
        tax_decimal = Decimal(str(tax))
        total_decimal = Decimal(str(total))
    except InvalidOperation:
        validation["status"] = "INVALID_NUMERIC_VALUES"
        return validation

    difference = abs(subtotal_decimal + tax_decimal - total_decimal)

    validation["difference"] = float(difference)
    validation["subtotal_plus_tax_matches_total"] = difference <= Decimal("0.02")
    validation["status"] = (
        "PASS"
        if difference <= Decimal("0.02")
        else "CHECK_REQUIRED"
    )

    return validation


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=" * 60)
    print("EXPERIMENT 15: MULTIMODAL RECEIPT EXTRACTION")
    print("=" * 60)

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the project .env file."
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Receipt image not found: {IMAGE_PATH}\n"
            "Place a receipt image in the images folder and name it receipt.jpg."
        )

    encoded_image, mime_type = encode_image(IMAGE_PATH)

    print(f"Image: {IMAGE_PATH}")
    print(f"Image MIME type: {mime_type}")
    print(f"Base64 image length: {len(encoded_image)} characters")
    print(f"Model: {MODEL}")
    print("\nSending receipt image and extraction prompt to Gemini...")

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=API_KEY,
        temperature=0,
        max_retries=1,
        timeout=60,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": RECEIPT_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_image}"
                },
            },
        ]
    )

    response = llm.invoke([message])
    response_text = response.content

    if isinstance(response_text, list):
        response_text = "\n".join(
            block.get("text", "")
            for block in response_text
            if isinstance(block, dict) and block.get("type") == "text"
        )

    receipt_data = parse_json_response(str(response_text))
    validation = validate_receipt(receipt_data)

    receipt_data["validation"] = validation

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(receipt_data, file, indent=2, ensure_ascii=False)

    with open(TEXT_OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("EXPERIMENT 15: MULTIMODAL RECEIPT EXTRACTION\n\n")
        file.write(f"Model: {MODEL}\n")
        file.write(f"Image: {IMAGE_PATH.name}\n\n")
        file.write("EXTRACTED JSON:\n")
        file.write(json.dumps(receipt_data, indent=2, ensure_ascii=False))
        file.write("\n")

    print("\n=== EXTRACTED RECEIPT JSON ===")
    print(json.dumps(receipt_data, indent=2, ensure_ascii=False))

    print("\n=== VALIDATION ===")
    print(f"Status: {validation['status']}")
    print(
        "Subtotal + tax matches total: "
        f"{validation['subtotal_plus_tax_matches_total']}"
    )

    print(f"\nJSON saved to: {OUTPUT_FILE}")
    print(f"Results saved to: {TEXT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()