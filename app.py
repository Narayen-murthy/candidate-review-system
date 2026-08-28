from flask import Flask, render_template, request, send_from_directory, jsonify
import os, json, time
from werkzeug.utils import secure_filename
from review_api import evaluate_with_job, compare_candidates

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def run_evaluate():
    candidate = request.form.get('candidate', 'A')
    candidate_name = request.form.get('candidate_name', 'Candidate').strip()
    job_desc = request.form.get('job_description', '').strip()
    use_sample = request.form.get('use_sample') == 'on'

    # get resume/transcript for candidate A
    def get_text(prefix):
        file_obj = request.files.get(f'{prefix}_file')
        if file_obj and file_obj.filename:
            fn = secure_filename(file_obj.filename)
            path = os.path.join(UPLOAD_DIR, fn)
            file_obj.save(path)
            return open(path, 'r', encoding='utf-8').read()
        return request.form.get(f'{prefix}_text','').strip()

    resume_text = get_text('resume')
    transcript_text = get_text('transcript')

    if use_sample:
        # evaluate_with_job will use SAMPLEs if empty
        pass

    if not resume_text and not transcript_text and use_sample:
        # review_api will substitute sample if needed
        pass

    if not resume_text or not transcript_text:
        return "Error: resume and transcript required (paste or upload), or check 'Use sample'", 400

    provider = 'openai' if request.form.get('real_llm') == 'on' else 'dummy'
    report = evaluate_with_job(resume_text, transcript_text, provider=provider, job_description=job_desc)

    ts = int(time.time())
    out_name = f"report_{candidate_name.replace(' ','_')}_{ts}.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    passed = (report.get('recommendation') == 'hire') or (report.get('confidence',0.0) >= float(request.form.get('pass_threshold',0.75)))

    return render_template('report.html', report=report, out_name=out_name, passed=passed)

@app.route('/compare', methods=['POST'])
def run_compare():
    job_desc = request.form.get('job_description', '').strip()
    provider = 'openai' if request.form.get('real_llm') == 'on' else 'dummy'

    # candidate A
    def get_text_for(prefix):
        file_obj = request.files.get(prefix + '_file')
        if file_obj and file_obj.filename:
            fn = secure_filename(file_obj.filename)
            path = os.path.join(UPLOAD_DIR, fn)
            file_obj.save(path)
            return open(path, 'r', encoding='utf-8').read()
        return request.form.get(prefix + '_text','').strip()

    r1 = get_text_for('resumeA')
    t1 = get_text_for('transcriptA')
    name1 = request.form.get('candidateA_name','CandidateA')

    r2 = get_text_for('resumeB')
    t2 = get_text_for('transcriptB')
    name2 = request.form.get('candidateB_name','CandidateB')

    if not ((r1 and t1) and (r2 and t2)):
        return "Error: both candidates require resume and transcript (paste or upload).", 400

    comparison = compare_candidates(r1, t1, r2, t2, provider=provider, job_description=job_desc)

    ts = int(time.time())
    out_name = f"compare_{ts}.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    return render_template('compare.html', comparison=comparison, out_name=out_name, name1=name1, name2=name2)

@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.json
    r = data.get('resume')
    t = data.get('transcript')
    provider = data.get('provider','dummy')
    job = data.get('job_description','')
    if not r or not t:
        return jsonify({'error':'resume and transcript required'}), 400
import os
from flask import Flask

app = Flask(__name__)

# your existing routes...

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
    return jsonify(evaluate_with_job(r, t, provider=provider, job_description=job))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
