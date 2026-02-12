import argparse
import base64
import json
from pathlib import Path

import requests


API_KEY = "AIzaSyAU-tGXw9isguA42NFlJ_YavlkDywUurrQ"
DEFAULT_MODEL = "gemini-3-flash"
ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a PDF to Gemini 3 Flash and print response text.")
    parser.add_argument("--pdf", required=True, help="Absolute or relative path to PDF file")
    parser.add_argument(
        "--prompt",
        default="Extract key fields from this PDF and return concise JSON.",
        help="Instruction prompt sent along with the PDF",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()

    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        return 1

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": args.prompt},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_b64,
                        }
                    },
                ]
            }
        ]
    }

    endpoint = f"{ENDPOINT_BASE}/{args.model}:generateContent"
    url = f"{endpoint}?key={API_KEY}"
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=240)

    print(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        print(response.text)
        return 1

    if response.status_code != 200:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 1

    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        print(text)
    except Exception:
        print(json.dumps(body, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
