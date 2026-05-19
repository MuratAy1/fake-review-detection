import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "review_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_INFO_PATH = MODELS_DIR / "label_map.json"

TEST_FILE = BASE_DIR / "external_test_set_50cg_50or_unique.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer bulunamadı: {VECTORIZER_PATH}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test dosyası bulunamadı: {TEST_FILE}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            model_info = json.load(f)
    else:
        model_info = {}

    df = pd.read_csv(TEST_FILE)

    print("Test dosyası:", TEST_FILE.name)
    print("Satır sayısı:", len(df))
    print("\nKolonlar:")
    print(df.columns.tolist())

    print("\nGerçek label dağılımı:")
    print(df["label"].value_counts())

    df["cleaned_text"] = df["text_"].apply(clean_text)

    X = vectorizer.transform(df["cleaned_text"])

    y_true = df["label"]
    y_pred_raw = model.predict(X)

    # Model 0/1 döndürüyorsa OR/CG'ye çevir.
    # Senin sisteminde genelde:
    # 0 = REAL / OR
    # 1 = FAKE / CG
    y_pred = []

    for pred in y_pred_raw:
        pred_str = str(pred)

        if pred_str == "0":
            y_pred.append("OR")
        elif pred_str == "1":
            y_pred.append("CG")
        elif pred_str.upper() in ["OR", "REAL"]:
            y_pred.append("OR")
        elif pred_str.upper() in ["CG", "FAKE"]:
            y_pred.append("CG")
        else:
            y_pred.append(pred_str)

    df["predicted_label"] = y_pred

    print("\nAccuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["FAKE / CG", "REAL / OR"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["OR", "CG"]))
    print("[[OR doğru, OR -> CG yanlış],")
    print(" [CG -> OR yanlış, CG doğru]]")

    cg_df = df[df["label"] == "CG"]
    or_df = df[df["label"] == "OR"]

    cg_correct = (cg_df["predicted_label"] == "CG").sum()
    or_correct = (or_df["predicted_label"] == "OR").sum()

    print("\nÖzet:")
    print(f"CG / fake yakalama: {cg_correct}/{len(cg_df)}")
    print(f"OR / real doğru tanıma: {or_correct}/{len(or_df)}")

    wrong_df = df[df["label"] != df["predicted_label"]].copy()

    print("\nYanlış tahmin sayısı:", len(wrong_df))

    if len(wrong_df) > 0:
        print("\n--- Yanlış tahminlerden ilk 20 örnek ---")
        for i, row in wrong_df.head(20).iterrows():
            print("\n" + "-" * 80)
            print("Gerçek label:", row["label"])
            print("Model tahmini:", row["predicted_label"])
            print("Kategori:", row.get("category", "N/A"))
            print("Yorum:")
            print(str(row["text_"])[:700])

    wrong_df.to_csv("external_test_wrong_predictions.csv", index=False, encoding="utf-8-sig")
    df.to_csv("external_test_predictions_full.csv", index=False, encoding="utf-8-sig")

    print("\nKaydedilen dosyalar:")
    print("external_test_predictions_full.csv")
    print("external_test_wrong_predictions.csv")


if __name__ == "__main__":
    main()
