from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    return "✅ AI Inventory Assistant is running."

@app.route('/chat', methods=['POST'])
def chat():
    """
    Generic chat with AI. No inventory data is considered.
    Payload: { user_message: "..." }
    """
    data = request.get_json()
    user_message = data.get('user_message', '').strip()

    if not user_message:
        return jsonify({'reply': '❌ No message received.'}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for general queries."},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response.choices[0].message.content
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'reply': f'⚠️ Error: {str(e)}'}), 500

@app.route('/ask-inventory', methods=['POST'])
def ask_inventory():
    """
    User sends a natural-language question with inventory data.
    AI responds based strictly on that data.
    Payload: {
      user_message,
      products,
      sales_orders,
      purchase_orders,
      suppliers,
      customers,
      locations,
      categories
    }
    """
    data = request.get_json()
    user_message = data.get('user_message', '').strip()

    if not user_message:
        return jsonify({'reply': '❌ Please include your question.'}), 400

    products = data.get('products', [])
    sales_orders = data.get('sales_orders', [])
    purchase_orders = data.get('purchase_orders', [])
    suppliers = data.get('suppliers', [])
    customers = data.get('customers', [])
    locations = data.get('locations', [])
    categories = data.get('categories', [])

    try:
        prompt = f"""
You are Logic, an AI Inventory Assistant. 
Answer the user's question strictly using the provided dataset. 
The response must be:
- Concise, clear, and data-driven.
- Well-structured (use Markdown tables, bullet points, or lists).
- Avoid vague or combined text blocks.
- If exact data is missing, say "I do not have enough information."

User Question: {user_message}

Dataset:
Products: {products}
Sales Orders: {sales_orders}
Purchase Orders: {purchase_orders}
Suppliers: {suppliers}
Customers: {customers}
Locations: {locations}
Categories: {categories}
"""

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are an intelligent assistant trained on structured inventory data."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f'⚠️ Error processing request: {str(e)}'}), 500

# No app.run()
