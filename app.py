import os
import re
import io
import math
import json
import joblib
import numpy as np
import pandas as pd
import tldextract
import google.generativeai as genai

from dotenv import load_dotenv
from collections import Counter
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# Load .env — key stays server-side, never sent to client
# ---------------------------------------------------------------------------
load_dotenv(override=True)

_AI_KEY = os.getenv('GEMINI_API_KEY', '').strip()
_AI_READY = False

if _AI_KEY:
    try:
        genai.configure(api_key=_AI_KEY)
        _AI_READY = True
        print('[OK] Secondary AI engine configured.')
    except Exception as e:
        print(f'[WARN] Secondary AI engine setup failed: {e}')
else:
    print('[WARN] Secondary AI engine key not found in .env')

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH    = os.path.join(MODELS_DIR, 'xgboost_model_use.pkl')
ENCODER_PATH  = os.path.join(MODELS_DIR, 'label_encoder_use.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_columns.json')
MAX_CSV_ROWS  = 5000

# ---------------------------------------------------------------------------
# Trusted domain whitelist
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS = {
    'google.com', 'googleapis.com', 'youtu.be', 'youtube.com',
    'gmail.com', 'googlemail.com', 'drive.google.com', 'docs.google.com',
    'maps.google.com', 'accounts.google.com',
    'microsoft.com', 'live.com', 'outlook.com', 'hotmail.com',
    'office.com', 'onedrive.com', 'sharepoint.com', 'azure.com', 'bing.com',
    'linkedin.com', 'facebook.com', 'instagram.com', 'whatsapp.com',
    'twitter.com', 'x.com', 'reddit.com', 'discord.com', 'telegram.org',
    'snapchat.com', 'tiktok.com', 'pinterest.com',
    'github.com', 'gitlab.com', 'stackoverflow.com', 'npmjs.com',
    'pypi.org', 'python.org', 'kaggle.com', 'anthropic.com', 'openai.com',
    'huggingface.co', 'wikipedia.org', 'bbc.com', 'bbc.co.uk', 'cnn.com',
    'reuters.com', 'nytimes.com', 'medium.com',
    'amazon.com', 'amazon.in', 'flipkart.com', 'ebay.com', 'paypal.com',
    'stripe.com', 'cloudflare.com', 'netlify.com', 'vercel.com',
}


def is_trusted_domain(url):
    try:
        ext = tldextract.extract(url)
        registered = f"{ext.domain}.{ext.suffix}".lower()
        if registered in TRUSTED_DOMAINS:
            return True
        parts = f"{ext.subdomain}.{registered}".strip('.').split('.')
        for i in range(len(parts) - 1):
            if '.'.join(parts[i:]) in TRUSTED_DOMAINS:
                return True
    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# Load ML model artifacts
# ---------------------------------------------------------------------------
try:
    model         = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_columns = json.load(f)
    print(f'[OK] ML model loaded        : {MODEL_PATH}')
    print(f'[OK] Label encoder loaded   : {ENCODER_PATH}')
    print(f'[OK] Feature columns loaded : {len(feature_columns)} features')
    print(f'[OK] Classes                : {label_encoder.classes_.tolist()}')
except Exception as e:
    print(f'[ERROR] Failed to load ML artifacts: {e}')
    model = label_encoder = feature_columns = None

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def calculate_entropy(s):
    if not s:
        return 0.0
    freq   = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_features(url):
    f = {}
    f['url_length']           = len(url)
    f['url_entropy']          = round(calculate_entropy(url), 4)
    f['has_https']            = int(url.startswith('https'))
    f['has_http']             = int(url.startswith('http://'))
    f['num_dots']             = url.count('.')
    f['num_hyphens']          = url.count('-')
    f['num_underscores']      = url.count('_')
    f['num_slashes']          = url.count('/')
    f['num_question_marks']   = url.count('?')
    f['num_equals']           = url.count('=')
    f['num_ampersands']       = url.count('&')
    f['num_at_symbols']       = url.count('@')
    f['num_percent']          = url.count('%')
    f['num_digits']           = sum(c.isdigit() for c in url)
    f['digit_ratio']          = round(f['num_digits'] / max(len(url), 1), 4)
    f['has_ip_address']       = int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url)))
    f['has_port']             = int(bool(re.search(r':\d{2,5}', url)))
    f['has_hex_encoding']     = int('%' in url)
    f['double_slash_in_path'] = int('//' in url[8:])
    f['has_shortening']       = int(bool(re.search(
        r'(bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly)',
        url, re.IGNORECASE)))
    try:
        ext       = tldextract.extract(url)
        domain    = ext.domain
        subdomain = ext.subdomain
        suffix    = ext.suffix
    except Exception:
        domain = subdomain = suffix = ''
    f['domain_length']    = len(domain)
    f['subdomain_count']  = len(subdomain.split('.')) if subdomain else 0
    f['domain_entropy']   = round(calculate_entropy(domain), 4)
    f['tld_length']       = len(suffix)
    return f


def build_feature_vector(url):
    features = extract_features(url)
    return np.array([[features.get(col, 0) for col in feature_columns]])

# ---------------------------------------------------------------------------
# Secondary AI analysis — internals never exposed to client
# ---------------------------------------------------------------------------

_AI_PROMPT = """You are a cybersecurity expert specializing in URL threat analysis.

Analyze the following URL and determine if it is safe or malicious.

URL: {url}

Initial analysis result: {ml_result}
Initial confidence: {ml_confidence}%
Detected attack type: {ml_attack}

Based on your knowledge of domain reputation, URL structure, phishing patterns,
TLD risk levels, and known malicious indicators, provide your expert analysis.

Respond ONLY in the following strict JSON format with no markdown or extra text:
{{
  "ai_verdict": "genuine" or "malicious",
  "ai_attack_type": null or one of ["phishing", "malware", "defacement", "spam"],
  "ai_confidence": a number 0-100,
  "ai_reasoning": "1-2 sentence explanation",
  "risk_indicators": ["list", "of", "red", "flags"] or [],
  "final_verdict": "genuine" or "malicious",
  "final_confidence": a number 0-100
}}"""


def _secondary_analysis(url, ml_result, ml_confidence, ml_attack):
    """
    Run secondary AI analysis server-side.
    The API key and model name never appear in any client response.
    Returns parsed dict or None on failure.
    """
    if not _AI_READY:
        return None
    try:
        ai_model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        prompt   = _AI_PROMPT.format(
            url=url,
            ml_result=ml_result,
            ml_confidence=ml_confidence,
            ml_attack=ml_attack or 'none'
        )
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            )
        )
        raw = response.text.strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'```$', '', raw).strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f'[WARN] Secondary analysis error: {e}')
        return None

# ---------------------------------------------------------------------------
# Combined prediction
# ---------------------------------------------------------------------------

def run_prediction(url):
    features_used  = extract_features(url)
    feature_subset = {k: features_used[k] for k in [
        'url_length', 'url_entropy', 'has_https', 'has_ip_address',
        'num_dots', 'has_shortening', 'num_at_symbols', 'has_port'
    ]}

    classes    = label_encoder.classes_.tolist() if label_encoder else []
    whitelisted= is_trusted_domain(url)

    # ML prediction
    if whitelisted:
        ml_result     = 'genuine'
        ml_confidence = 100.0
        ml_attack     = None
        class_proba   = {cls: (100.0 if cls == 'benign' else 0.0) for cls in classes}
    else:
        if model is None:
            return {'error': 'ML model not loaded.'}, 503
        vec        = build_feature_vector(url)
        pred_idx   = int(model.predict(vec)[0])
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        proba      = model.predict_proba(vec)[0]
        ml_result     = 'malicious' if pred_label.lower() != 'benign' else 'genuine'
        ml_confidence = round(float(proba[pred_idx]) * 100, 2)
        ml_attack     = pred_label if ml_result == 'malicious' else None
        class_proba   = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}

    # Secondary AI analysis
    ai = _secondary_analysis(url, ml_result, ml_confidence, ml_attack)

    # Final verdict
    if ai:
        final_verdict    = ai.get('final_verdict', ml_result)
        final_confidence = ai.get('final_confidence', ml_confidence)
        final_attack     = ai.get('ai_attack_type', ml_attack) if final_verdict == 'malicious' else None
    else:
        final_verdict    = ml_result
        final_confidence = ml_confidence
        final_attack     = ml_attack

    # Build response — NO api key, NO model name, NO vendor name exposed
    return {
        'url'         : url,
        'result'      : final_verdict,
        'attack_type' : final_attack,
        'confidence'  : final_confidence,
        'whitelisted' : whitelisted,

        'ml': {
            'verdict'    : ml_result,
            'confidence' : ml_confidence,
            'attack_type': ml_attack,
        },

        # "ai" key only — no vendor name in response
        'ai': {
            'available'      : ai is not None,
            'verdict'        : ai.get('ai_verdict')        if ai else None,
            'confidence'     : ai.get('ai_confidence')     if ai else None,
            'attack_type'    : ai.get('ai_attack_type')    if ai else None,
            'reasoning'      : ai.get('ai_reasoning')      if ai else None,
            'risk_indicators': ai.get('risk_indicators', [])if ai else [],
        },

        'class_probabilities': class_proba,
        'features'           : feature_subset,
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/analyze')
def analyze():
    return render_template('analyze.html')


@app.route('/api/predict-url', methods=['POST'])
def predict_url():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing field: url'}), 400
    url = str(data['url']).strip()
    if not url:
        return jsonify({'error': 'URL cannot be empty.'}), 400
    if len(url) > 2048:
        return jsonify({'error': 'URL too long (max 2048 chars).'}), 400
    try:
        result = run_prediction(url)
        if isinstance(result, tuple):
            return jsonify({'error': result[0]}), result[1]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict-csv', methods=['POST'])
def predict_csv():
    if model is None:
        return jsonify({'error': 'ML model not loaded.'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    uploaded_file = request.files['file']
    if not uploaded_file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files accepted.'}), 400
    try:
        content = uploaded_file.read().decode('utf-8', errors='replace')
        df      = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400

    url_col = None
    for candidate in ['url', 'URL', 'Url', 'link', 'Link']:
        if candidate in df.columns:
            url_col = candidate
            break
    if url_col is None:
        return jsonify({'error': f'No URL column found. Columns: {df.columns.tolist()}'}), 400
    if len(df) > MAX_CSV_ROWS:
        return jsonify({'error': f'CSV exceeds {MAX_CSV_ROWS} row limit.'}), 400

    results = []
    malicious = genuine = 0
    type_count = {}

    for url in df[url_col].fillna('').astype(str):
        url = url.strip()
        if not url:
            continue
        try:
            r = run_prediction(url)
            if isinstance(r, tuple):
                continue
            results.append(r)
            if r['result'] == 'malicious':
                malicious += 1
                t = r.get('attack_type') or 'unknown'
                type_count[t] = type_count.get(t, 0) + 1
            else:
                genuine += 1
        except Exception as e:
            results.append({'url': url, 'result': 'error', 'error': str(e)})

    total = malicious + genuine
    return jsonify({
        'total'        : total,
        'malicious'    : malicious,
        'genuine'      : genuine,
        'malicious_pct': round(malicious / max(total, 1) * 100, 1),
        'attack_types' : type_count,
        'results'      : results,
    }), 200


@app.route('/api/status', methods=['GET'])
def status():
    # Returns capability flags only — no vendor names, no key info
    return jsonify({
        'status'          : 'ok' if model is not None else 'model_not_loaded',
        'ml_ready'        : model is not None,
        'ai_ready'        : _AI_READY,
        'classes'         : label_encoder.classes_.tolist() if label_encoder else [],
        'trusted_domains' : len(TRUSTED_DOMAINS),
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
