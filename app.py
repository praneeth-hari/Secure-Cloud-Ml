"""SecureCloudML Flask application.

Serves the model-comparison dashboard and a CSV-upload prediction page
(session-authenticated), plus a JWT-protected JSON API for programmatic
predictions. Every request that carries row data through `/PredictAction`
is AES-256-CBC encrypted before being handed to the ML pipeline, and
decrypted server-side, to demonstrate the secure-transmission story this
project is named for.
"""
from __future__ import annotations

import hmac
import io
import json
import logging
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)

import config
import train
from encryption import decrypt_data, encrypt_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["JWT_SECRET_KEY"] = config.JWT_SECRET_KEY
jwt = JWTManager(app)

logger.info("Ensuring trained model artifacts are available...")
train.ensure_trained()
ARTIFACTS = train.load_artifacts()
MODEL_ORDER = ["KNN", "ANN", "ANN with ABC", "Random Forest", "XGBoost"]
NUM_FEATURES = ARTIFACTS["scaler"].n_features_in_


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def _valid_credentials(username: str, password: str) -> bool:
    return hmac.compare_digest(username, config.API_USERNAME) and hmac.compare_digest(
        password, config.API_PASSWORD
    )


def _encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved label encoders to every categorical column present in df."""
    df = df.copy()
    for column_name, encoder in ARTIFACTS["label_encoders"]:
        if column_name in df.columns:
            df[column_name] = encoder.transform(df[column_name].astype(str))
    df.fillna(0, inplace=True)
    return df


def _predict_all_models(x_scaled: np.ndarray) -> dict[str, np.ndarray]:
    """Run every trained model on already-scaled features and return predicted class indices."""
    predictions: dict[str, np.ndarray] = {}
    for name, model in ARTIFACTS["models"].items():
        if name in ("ANN", "ANN with ABC"):
            features = x_scaled[:, ARTIFACTS["best_features"]] if name == "ANN with ABC" else x_scaled
            predictions[name] = np.argmax(model.predict(features, verbose=0), axis=1)
        else:
            predictions[name] = model.predict(x_scaled)
    return predictions


def _labels_for(predictions: np.ndarray) -> list[str]:
    return [str(ARTIFACTS["labels"][i]) for i in predictions]


# --------------------------------------------------------------------------
# HTML routes (session-authenticated)
# --------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if _valid_credentials(username, password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password."), 401
    return render_template("login.html")


@app.route("/Logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@app.route("/index", methods=["GET"])
@login_required
def index():
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    chart_data = {
        "labels": list(metrics.keys()),
        "accuracy": [round(metrics[m]["accuracy"], 2) for m in metrics],
        "precision": [round(metrics[m]["precision"], 2) for m in metrics],
        "recall": [round(metrics[m]["recall"], 2) for m in metrics],
        "fscore": [round(metrics[m]["fscore"], 2) for m in metrics],
    }
    return render_template("index.html", data="", chart_data=chart_data)


@app.route("/Predict", methods=["GET"])
@login_required
def predict_page():
    return render_template("Predict.html")


@app.route("/PredictAction", methods=["POST"])
@login_required
def predict_action():
    uploaded = request.files.get("t1")
    if uploaded and uploaded.filename:
        raw_csv_bytes = uploaded.read()
    else:
        with open("Dataset/testData.csv", "rb") as f:
            raw_csv_bytes = f.read()

    # Simulate a secure transmission channel: encrypt the payload, then
    # decrypt it server-side before it ever reaches the ML pipeline.
    encrypted_payload = encrypt_data(raw_csv_bytes.decode("utf-8", errors="replace"))
    decrypted_payload = decrypt_data(encrypted_payload)

    test_dataset = pd.read_csv(io.StringIO(decrypted_payload))

    # Drop target column if uploaded CSV already contains labels
    features_df = test_dataset.drop(columns=["Attack_Type"]) if "Attack_Type" in test_dataset.columns else test_dataset.copy()

    encoded = _encode_dataframe(features_df)
    x_scaled = ARTIFACTS["scaler"].transform(encoded.to_numpy())
    predictions = _predict_all_models(x_scaled)
    labels_by_model = {m: _labels_for(predictions[m]) for m in MODEL_ORDER}

    rows_html = []
    for i in range(len(test_dataset)):
        cells = "".join(
            f'<td><span class="badge {"normal" if labels_by_model[m][i] == "normal" else "attack"}">'
            f'{labels_by_model[m][i]}</span></td>'
            for m in MODEL_ORDER
        )
        row = test_dataset.iloc[i]
        proto = row.get("protocol_type", "-")
        svc = row.get("service", "-")
        flag = row.get("flag", "-")
        src_b = row.get("src_bytes", 0)
        dst_b = row.get("dst_bytes", 0)
        desc = (
            f"<strong>Sample #{i+1}</strong> &mdash; "
            f"Proto: <code>{proto}</code>, Service: <code>{svc}</code>, Flag: <code>{flag}</code>, "
            f"Bytes: <code>{src_b}/{dst_b}</code>"
        )
        rows_html.append(f"<tr><td style='text-align:left;'>{desc}</td>{cells}</tr>")

    header_cells = "".join(f"<th>{m}</th>" for m in MODEL_ORDER)
    output = (
        '<table class="result-table">'
        f"<thead><tr><th style='text-align:left;'>Network Traffic Sample</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )

    with open("model/metrics.json") as f:
        metrics = json.load(f)
    chart_data = {
        "labels": list(metrics.keys()),
        "accuracy": [round(metrics[m]["accuracy"], 2) for m in metrics],
        "precision": [round(metrics[m]["precision"], 2) for m in metrics],
        "recall": [round(metrics[m]["recall"], 2) for m in metrics],
        "fscore": [round(metrics[m]["fscore"], 2) for m in metrics],
    }

    return render_template("index.html", data=output, chart_data=chart_data)


# --------------------------------------------------------------------------
# JSON REST API (JWT-authenticated)
# --------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify(status="ok")


@app.route("/api/login", methods=["POST"])
def api_login():
    body = request.get_json(silent=True) or {}
    username = body.get("username", "")
    password = body.get("password", "")
    if not _valid_credentials(username, password):
        return jsonify(error="Invalid credentials"), 401
    token = create_access_token(identity=username)
    return jsonify(access_token=token)


@app.route("/api/predict", methods=["POST"])
@jwt_required()
def api_predict():
    body = request.get_json(silent=True) or {}
    rows = body.get("data")
    if not isinstance(rows, list) or not rows:
        return jsonify(error="'data' must be a non-empty list of feature rows"), 400
    if any(not isinstance(row, list) or len(row) != NUM_FEATURES for row in rows):
        return jsonify(error=f"each row must have exactly {NUM_FEATURES} numeric features"), 400

    try:
        x = np.array(rows, dtype=float)
    except (TypeError, ValueError):
        return jsonify(error="all feature values must be numeric"), 400

    x_scaled = ARTIFACTS["scaler"].transform(x)
    predictions = _predict_all_models(x_scaled)
    labels_by_model = {m: _labels_for(predictions[m]) for m in MODEL_ORDER}

    results = [
        {model: labels_by_model[model][i] for model in MODEL_ORDER}
        for i in range(len(rows))
    ]

    return jsonify(requested_by=get_jwt_identity(), predictions=results)


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="Not found"), 404
    return render_template("login.html", error="Page not found."), 404


@app.errorhandler(500)
def server_error(error):
    logger.exception("Unhandled server error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify(error="Internal server error"), 500
    return render_template("login.html", error="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=config.FLASK_ENV != "production")
