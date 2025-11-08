from flask import Flask, request, jsonify
import os
from prototype9 import analyze_image  # refactor your code into a function

app = Flask(__name__)

@app.route("/")
def home():
    return "Face & Hair Analyzer API is running."

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filename)

    # Call your analysis function (refactor your prototype9.py)
    result = analyze_image(filename)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
