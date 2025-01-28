from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import spacy
import requests

# Initialize Flask app
app = Flask(__name__)

# Load NLP and semantic similarity model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Google Fact Check API key (replace with your API key)
FACT_CHECK_API_KEY = "API_KEY"

# Helper: Extract claims from text
def extract_claims(text):
    doc = nlp(text)
    claims = [sent.text for sent in doc.sents if len(sent.text.strip()) > 10]
    return claims

# Helper: Query fact-checking API
def query_fact_check_api(claim):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": claim, "key": FACT_CHECK_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("claims", [])
    return []

# Helper: Validate claim using semantic similarity
def validate_claim(claim):
    sources = query_fact_check_api(claim)
    if not sources:
        return {"status": "Unverified", "evidence": []}

    # Extract evidence texts
    evidence_texts = [source["text"] for source in sources]
    claim_embedding = model.encode(claim)
    evidence_embeddings = model.encode(evidence_texts)

    # Compute similarity scores
    similarities = util.cos_sim(claim_embedding, evidence_embeddings).squeeze()
    max_similarity = similarities.max().item()

    # Classification based on similarity
    if max_similarity > 0.8:
        return {"status": "True", "evidence": evidence_texts}
    elif max_similarity > 0.5:
        return {"status": "Possibly True", "evidence": evidence_texts}
    else:
        return {"status": "False", "evidence": evidence_texts}

# Flask route: Fact-checking endpoint
@app.route("/fact-check", methods=["POST"])
def fact_check():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    # Extract claims and validate them
    text = data["text"]
    claims = extract_claims(text)
    results = [{"claim": claim, "result": validate_claim(claim)} for claim in claims]

    return jsonify(results)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
