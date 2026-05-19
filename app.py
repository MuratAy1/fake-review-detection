from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import nltk
import numpy as np
import streamlit as st

from fake_review_detector_4 import preprocess_text

st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "review_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

UNCERTAIN_CONFIDENCE_THRESHOLD = 0.55
UNCERTAIN_RAW_MARGIN = 0.20

for resource in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    nltk.download(resource, quiet=True)


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_artifacts():
    missing = [str(p) for p in (MODEL_PATH, VECTORIZER_PATH, LABEL_MAP_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Eksik dosyalar: " + ", ".join(missing))

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_info = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return model, vectorizer, label_info


def get_english_label(label_info: dict, pred: int) -> str:
    pred_key = str(pred)

    if pred_key in label_info:
        return label_info[pred_key]

    reverse_map = label_info.get("reverse_label_map", {})
    if pred_key in reverse_map:
        return reverse_map[pred_key]

    return "FAKE" if pred == 1 else "REAL"


def get_turkish_label(label_info: dict, english_label: str) -> str:
    turkish_labels = label_info.get(
        "turkish_labels",
        {
            "REAL": "GERÇEK YORUM",
            "FAKE": "SAHTE YORUM",
        },
    )
    return turkish_labels.get(english_label, english_label)


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + np.exp(-x))


def calculate_confidence(model, features, pred: int) -> tuple[float, dict]:
    debug = {
        "method": None,
        "classes": getattr(model, "classes_", None).tolist()
        if hasattr(getattr(model, "classes_", None), "tolist")
        else getattr(model, "classes_", None),
        "raw_score": None,
        "probabilities": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        class_index = classes.index(pred)

        debug["method"] = "predict_proba"
        debug["probabilities"] = probabilities.tolist()

        return float(probabilities[class_index]), debug

    if hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(features)[0])
        classes = list(model.classes_)
        positive_class = int(classes[1])
        positive_confidence = sigmoid(raw_score)

        if pred == positive_class:
            confidence = positive_confidence
        else:
            confidence = 1.0 - positive_confidence

        debug["method"] = "decision_function + sigmoid"
        debug["raw_score"] = raw_score
        debug["positive_class"] = positive_class
        debug["positive_confidence"] = positive_confidence

        return float(confidence), debug

    return 0.0, debug


def predict_review(review_text: str):
    model, vectorizer, label_info = load_artifacts()

    cleaned = preprocess_text(review_text)
    if not cleaned:
        return None, None, cleaned, None

    features = vectorizer.transform([cleaned])
    pred = int(model.predict(features)[0])
    confidence, debug = calculate_confidence(model, features, pred)

    english_label = get_english_label(label_info, pred)
    turkish_label = get_turkish_label(label_info, english_label)

    debug["raw_prediction"] = pred
    debug["english_label"] = english_label
    debug["turkish_label"] = turkish_label

    return english_label, confidence, cleaned, debug


def is_uncertain_prediction(confidence: float, debug: dict | None) -> bool:
    if confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
        return True

    if debug:
        raw_score = debug.get("raw_score")
        if raw_score is not None and abs(float(raw_score)) < UNCERTAIN_RAW_MARGIN:
            return True

    return False


if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def clear_input():
    st.session_state.review_text = ""
    st.session_state.last_result = None


st.title("Fake Review Detection System")
st.caption("Review authenticity analysis interface")

st.info(
    "This model was trained on an English fake review dataset. "
    "For reliable results, enter English product reviews."
)

try:
    model, vectorizer, label_info = load_artifacts()
except FileNotFoundError as exc:
    st.error("Model dosyaları bulunamadı.")
    st.code(str(exc), language="text")
    st.markdown(
        "Önce proje klasöründe şu komutu çalıştır:\n\n"
        "```bash\npython train_and_save_model.py\n```"
    )
    st.stop()

st.text_area(
    "Enter a product review:",
    key="review_text",
    height=180,
    placeholder="Example: This product is amazing, the quality is perfect and I recommend it to everyone...",
)

col1, col2 = st.columns(2)
analyze_clicked = col1.button("Analyze", type="primary", use_container_width=True)
col2.button("Clear", on_click=clear_input, use_container_width=True)

if analyze_clicked:
    review = st.session_state.review_text.strip()
    if not review:
        st.warning("Please enter a review first.")
    else:
        start = time.perf_counter()
        label, confidence, cleaned, debug = predict_review(review)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if label is None:
            st.warning("The text became empty after preprocessing. Please enter a more descriptive review.")
        else:
            st.session_state.last_result = {
                "label": label,
                "confidence": confidence,
                "cleaned": cleaned,
                "debug": debug,
                "elapsed_ms": elapsed_ms,
            }

if st.session_state.last_result:
    result = st.session_state.last_result
    label = result["label"]
    confidence = result["confidence"]
    elapsed_ms = result.get("elapsed_ms", 0.0)

    st.divider()
    st.subheader("Result")

    is_uncertain = is_uncertain_prediction(confidence, result.get("debug"))
    display_label = "UNCERTAIN" if is_uncertain else label

    if is_uncertain:
        st.warning("⚠️ The model is **UNCERTAIN** for this review.")
        st.caption(f"Raw model prediction: {label}")
    elif label == "FAKE":
        st.error("⚠️ This review is classified as **FAKE**.")
    else:
        st.success("✅ This review is classified as **REAL**.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", display_label)
    m2.metric("Confidence Score", f"{confidence:.1%}")
    m3.metric("Prediction time", f"{elapsed_ms:.1f} ms")
    st.progress(int(confidence * 100))
