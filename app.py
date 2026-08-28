import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from candidate_review import evaluate, SAMPLE_RESUME, SAMPLE_TRANSCRIPT


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_uploaded_file(file_obj):
    """Read an uploaded text file safely."""
    if not file_obj or not file_obj.filename:
        return ""

    filename = secure_filename(file_obj.filename)

    # Only allow text-like files
    allowed = {
        ".txt",
        ".md",
        ".csv",
        ".json",
    }

    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed:
        raise ValueError(
            "Only TXT, MD, CSV, and JSON files are supported."
        )

    path = os.path.join(UPLOAD_DIR, filename)
    file_obj.save(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def get_text(form_name):
    """
    Get text either from an uploaded file or textarea.
    """
    file_obj = request.files.get(f"{form_name}_file")

    if file_obj and file_obj.filename:
        return read_uploaded_file(file_obj)

    return request.form.get(f"{form_name}_text", "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def run_evaluation():

    candidate_name = (
        request.form.get("candidate_name", "Candidate")
        .strip()
        or "Candidate"
    )

    job_description = request.form.get(
        "job_description", ""
    ).strip()

    use_sample = request.form.get("use_sample") == "on"

    try:
        resume_text = get_text("resume")
        transcript_text = get_text("transcript")

        # Dummy sample mode
        if use_sample:
            if not resume_text:
                resume_text = SAMPLE_RESUME

            if not transcript_text:
                transcript_text = SAMPLE_TRANSCRIPT

        if not resume_text:
            return render_template(
                "index.html",
                error="Please provide a resume or select 'Use sample candidate'.",
            ), 400

        if not transcript_text:
            return render_template(
                "index.html",
                error="Please provide an interview transcript or select 'Use sample candidate'.",
            ), 400

        # Force dummy LLM
        report = evaluate(
            resume_text,
            transcript_text,
            provider="dummy",
        )

        # Save report
        timestamp = int(time.time())

        safe_name = secure_filename(candidate_name)

        if not safe_name:
            safe_name = "candidate"

        filename = f"report_{safe_name}_{timestamp}.json"

        output_path = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return render_template(
            "report.html",
            report=report,
            candidate_name=candidate_name,
            job_description=job_description,
            output_file=filename,
        )

    except Exception as exc:

        return render_template(
            "index.html",
            error=f"Evaluation failed: {str(exc)}",
        ), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():

    data = request.get_json(silent=True) or {}

    resume = data.get("resume", "")
    transcript = data.get("transcript", "")

    if not resume or not transcript:
        return jsonify({
            "error": "resume and transcript are required"
        }), 400

    try:

        report = evaluate(
            resume,
            transcript,
            provider="dummy",
        )

        return jsonify(report)

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500


@app.route("/outputs/<path:filename>")
def download_output(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=True,
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "candidate-review-system",
        "llm": "dummy",
    })


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8080,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )

