from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from chroma_helper import ChromaHelper
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chroma = ChromaHelper()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    success, message = chroma.ingest_csv(filepath)
    if success:
        return jsonify({"message": "File uploaded and processed successfully"})
    else:
        return jsonify({"error": message}), 500

@app.route("/api/query", methods=["POST"])
def handle_query():
    question = request.json.get("question", "")
    n_results = request.json.get("limit", 50)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    relevant_rows = chroma.search(question, n_results=n_results)

    prompt = f"""You are a helpful data assistant. Respond only based on the following rows.

Question: {question}

Relevant CSV Data (max {n_results} rows):
{relevant_rows}

Answer clearly. If the question is about a count, count the matching rows exactly."""

    response = chroma.call_openai(prompt)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
