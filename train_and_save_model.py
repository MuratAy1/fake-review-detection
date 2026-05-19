# train_and_save_model.py

import os
import re
import json
import string
import warnings
import joblib

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


DATA_FILE = "fake_reviews_dataset.csv"
TEXT_COLUMN = "text_"
LABEL_COLUMN = "label"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "review_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
INFO_FILE = os.path.join(MODEL_DIR, "label_map.json")

RANDOM_STATE = 42

for resource in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    nltk.download(resource, quiet=True)


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)


def load_and_prepare_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} bulunamadı. CSV dosyasını bu .py dosyasıyla aynı klasöre koy."
        )

    df = pd.read_csv(DATA_FILE)

    print("Veri yüklendi.")
    print("Satır / sütun:", df.shape)
    print("Kolonlar:", list(df.columns))

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"'{TEXT_COLUMN}' kolonu bulunamadı.")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"'{LABEL_COLUMN}' kolonu bulunamadı.")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

    print("\nEtiket dağılımı:")
    print(df[LABEL_COLUMN].value_counts())

    print("\nMetinler temizleniyor...")
    df["cleaned_text"] = df[TEXT_COLUMN].apply(preprocess_text)

    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)


    label_map = {
        "OR": 0,
        "CG": 1
    }

    df["label_numeric"] = df[LABEL_COLUMN].map(label_map)

    if df["label_numeric"].isnull().any():
        raise ValueError(
            "Etiketlerde OR / CG dışında değer var. Dataset label kolonunu kontrol et."
        )

    print("\nTemizlik sonrası veri:", df.shape)
    print("Sayısal etiket dağılımı:")
    print(df["label_numeric"].value_counts())

    return df, label_map

def create_tfidf():
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
        sublinear_tf=True,
    )

def create_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="liblinear",
            C=1.0
        ),

        "SVM LinearSVC": LinearSVC(
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            C=1.0
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),
    }

def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df, label_map = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label_numeric"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label_numeric"]
    )

    print("\nTrain:", X_train.shape)
    print("Test:", X_test.shape)

    tfidf = create_tfidf()

    print("\nTF-IDF hazırlanıyor...")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("TF-IDF train shape:", X_train_tfidf.shape)
    print("TF-IDF test shape:", X_test_tfidf.shape)

    models = create_models()
    results = {}

    for name, model in models.items():
        print("\n" + "-" * 60)
        print(f"Model eğitiliyor: {name}")
        print("-" * 60)

        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        cv_scores = cross_val_score(
            model,
            X_train_tfidf,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"CV Mean  : {cv_scores.mean():.4f}")
        print(f"CV Std   : {cv_scores.std():.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=["REAL / OR", "FAKE / CG"]
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = results[best_model_name]["model"]

    print("\n" + "=" * 60)
    print("EN İYİ MODEL")
    print("=" * 60)
    print("Model:", best_model_name)
    print("F1:", results[best_model_name]["f1"])
    print("Accuracy:", results[best_model_name]["accuracy"])

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(tfidf, VECTORIZER_FILE)

    model_info = {
    "0": "REAL",
    "1": "FAKE",

    "label_map": {
        "OR": 0,
        "CG": 1
    },

    "reverse_label_map": {
        "0": "REAL",
        "1": "FAKE"
    },

    "source_labels": {
        "OR": "REAL",
        "CG": "FAKE"
    },

    "turkish_labels": {
        "REAL": "GERÇEK YORUM",
        "FAKE": "SAHTE YORUM"
    },

    "best_model_name": best_model_name,

    "model_classes": best_model.classes_.tolist() if hasattr(best_model, "classes_") else None,

    "tfidf_params": {
        "max_features": 20000,
        "ngram_range": [1, 2],
        "max_df": 0.95,
        "min_df": 2,
        "sublinear_tf": True
    },

    "model_results": {
        name: {
            "accuracy": results[name]["accuracy"],
            "precision": results[name]["precision"],
            "recall": results[name]["recall"],
            "f1": results[name]["f1"],
            "cv_mean": results[name]["cv_mean"],
            "cv_std": results[name]["cv_std"],
        }
        for name in results
    }
}

    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    print("\nModel kaydedildi:")
    print(MODEL_FILE)
    print(VECTORIZER_FILE)
    print(INFO_FILE)

    print("\nBitti.")


if __name__ == "__main__":
    train_and_save()
