from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# ==========================
# Setup
# ==========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    return "✅ AI Inventory Assistant is running."


# ==========================
# PROMPT BUILDER
# ==========================
def build_prompt(user_message: str, context: str = "", mode: str = "general") -> str:
    """
    Build standardized prompt for AI.
    """
    base_schema = """
⚠️ IMPORTANT RESPONSE RULES:
- Always respond in **valid JSON** only.
- Schema must be:

{
  "sender": "bot",
  "title": "string (short descriptive heading)",
  "sections": [
    {
      "type": "list" | "table" | "note" | "text",
      "headers": [optional for table],
      "rows": [optional for table],
      "data": [optional for list],
      "text": "optional for note/text"
    }
  ]
}

- Do NOT wrap JSON in code fences or text.
"""

    if mode == "general":
        return f"""
You are Logic, a helpful AI assistant for general queries.

{base_schema}

User Question: {user_message}
"""
    elif mode == "inventory":
        return f"""
You are Logic, an AI Inventory Assistant.
Answer the user's question strictly using the provided dataset.

{base_schema}

- If data is missing, return:

{{
  "sender": "bot",
  "title": "⚠️ Not Enough Data",
  "sections": [
    {{
      "type": "note",
      "text": "I do not have enough information."
    }}
  ]
}}

User Question: {user_message}

Dataset:
{context}
"""


# ==========================
# OPENAI CALLER
# ==========================
def call_openai(prompt: str) -> dict:
    """
    Call OpenAI with error handling. Always return JSON.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        reply = response.choices[0].message.content.strip()

        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            return {
                "sender": "bot",
                "title": "⚠️ Response Format Error",
                "sections": [
                    {"type": "note", "text": "AI returned an invalid format."}
                ]
            }

    except Exception as e:
        return {
            "sender": "bot",
            "title": "⚠️ Server Error",
            "sections": [
                {"type": "note", "text": str(e)}
            ]
        }


# ==========================
# ROUTES
# ==========================
@app.route('/chat', methods=['POST'])
def chat():
    """
    General AI chat.
    """
    data = request.get_json()
    user_message = data.get('user_message', '').strip()

    if not user_message:
        return jsonify({
            "sender": "bot",
            "title": "⚠️ Input Error",
            "sections": [
                {"type": "note", "text": "❌ No message received."}
            ]
        }), 400

    prompt = build_prompt(user_message, mode="general")
    reply_json = call_openai(prompt)
    return jsonify(reply_json)


@app.route('/ask-inventory', methods=['POST'])
def ask_inventory():
    """
    Inventory-specific AI chat.
    """
    data = request.get_json()
    user_message = data.get('user_message', '').strip()

    if not user_message:
        return jsonify({
            "sender": "bot",
            "title": "⚠️ Input Error",
            "sections": [
                {"type": "note", "text": "❌ Please include your question."}
            ]
        }), 400

    # Dataset context
    context = json.dumps({
        "Products": data.get('products', []),
        "Sales Orders": data.get('sales_orders', []),
        "Purchase Orders": data.get('purchase_orders', []),
        "Suppliers": data.get('suppliers', []),
        "Customers": data.get('customers', []),
        "Locations": data.get('locations', []),
        "Categories": data.get('categories', [])
    }, indent=2)

    prompt = build_prompt(user_message, context=context, mode="inventory")
    reply_json = call_openai(prompt)
    return jsonify(reply_json)


# ==========================
# Production: Use gunicorn/vercel
# ==========================
# No app.run() here
