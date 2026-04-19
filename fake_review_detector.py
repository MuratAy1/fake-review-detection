"""
Fake Reviews and Opinion Spam Detection using Machine Learning Algorithms.

Pipeline compares four supervised classifiers (Naive Bayes, Logistic Regression,
LinearSVC, Random Forest) on the Kaggle Fake Reviews Dataset using TF-IDF
features over cleaned review text. Outputs evaluation metrics, confusion
matrices and cross-validation scores as PNG figures and a CSV summary.

Dataset: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
"""

import os
import re
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Download required NLTK resources (silent if already present).
for resource in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    nltk.download(resource, quiet=True)

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data loading and exploration
# ---------------------------------------------------------------------------
def load_data(filepath):
    """Load the dataset and print basic information about its structure."""
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df


def explore_data(df, text_column, label_column):
    """Generate exploratory plots: class distribution, length distributions, word clouds."""
    print("\n" + "=" * 60)
    print("Step 2: Exploratory data analysis")
    print("=" * 60)

    print(f"\nClass distribution:\n{df[label_column].value_counts()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # Class distribution
    class_counts = df[label_column].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0, 0].bar(class_counts.index, class_counts.values, color=colors)
    axes[0, 0].set_title("Class distribution (Fake vs Real)")
    axes[0, 0].set_xlabel("Label")
    axes[0, 0].set_ylabel("Number of reviews")
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + 200, str(v), ha="center", fontweight="bold")

    # Length features
    df["review_length"] = df[text_column].astype(str).apply(len)
    df["word_count"] = df[text_column].astype(str).apply(lambda x: len(x.split()))

    # Character length distribution
    for label in df[label_column].unique():
        subset = df[df[label_column] == label]
        axes[0, 1].hist(subset["review_length"], bins=50, alpha=0.6, label=label)
    axes[0, 1].set_title("Review length (characters)")
    axes[0, 1].set_xlabel("Character count")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    # Word count distribution
    for label in df[label_column].unique():
        subset = df[df[label_column] == label]
        axes[1, 0].hist(subset["word_count"], bins=50, alpha=0.6, label=label)
    axes[1, 0].set_title("Word count distribution")
    axes[1, 0].set_xlabel("Number of words")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    # Mean word count by class
    stats = df.groupby(label_column)["word_count"].mean()
    axes[1, 1].bar(stats.index, stats.values, color=colors)
    axes[1, 1].set_title("Average word count per class")
    axes[1, 1].set_xlabel("Label")
    axes[1, 1].set_ylabel("Mean word count")

    plt.tight_layout()
    plt.savefig("01_eda_analizi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 01_eda_analizi.png")

    if WORDCLOUD_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, label in enumerate(df[label_column].unique()):
            text = " ".join(df[df[label_column] == label][text_column].astype(str).tolist())
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                colormap="viridis" if idx == 0 else "magma",
            ).generate(text)
            axes[idx].imshow(wc, interpolation="bilinear")
            axes[idx].set_title(f"Word cloud: {label}", fontsize=14)
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig("02_kelime_bulutu.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved 02_kelime_bulutu.png")

    return df


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Clean a single review: lowercase, strip URLs/HTML/digits/punctuation,
    remove English stopwords, lemmatize, drop one-character tokens."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
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


def preprocess_dataframe(df, text_column, label_column):
    """Apply preprocess_text to every row and drop empties."""
    print("\n" + "=" * 60)
    print("Step 3: Text preprocessing")
    print("=" * 60)

    print("Cleaning text (this may take a few minutes)...")
    df["cleaned_text"] = df[text_column].apply(preprocess_text)

    empty_count = (df["cleaned_text"] == "").sum()
    if empty_count > 0:
        print(f"Dropping {empty_count} rows that became empty after cleaning.")
        df = df[df["cleaned_text"] != ""].reset_index(drop=True)

    print(f"Preprocessing complete. Remaining rows: {len(df)}")

    print("\nBefore/After example:")
    print(f"  Raw:     {df[text_column].iloc[0][:100]}...")
    print(f"  Cleaned: {df['cleaned_text'].iloc[0][:100]}...")

    return df


# ---------------------------------------------------------------------------
# Feature extraction (TF-IDF)
# ---------------------------------------------------------------------------
def extract_features(df, label_column, max_features=10000):
    """Convert cleaned text to TF-IDF features and return train/test splits."""
    print("\n" + "=" * 60)
    print("Step 4: Feature extraction (TF-IDF)")
    print("=" * 60)

    # Map string labels to binary targets. CG = fake (1), OR = real (0).
    if df[label_column].dtype == "object":
        unique_labels = df[label_column].unique()
        print(f"Found labels: {unique_labels}")
        if "CG" in unique_labels and "OR" in unique_labels:
            label_map = {"CG": 1, "OR": 0}
        else:
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
        df["label_numeric"] = df[label_column].map(label_map)
        print(f"Label mapping: {label_map}")
    else:
        label_map = None
        df["label_numeric"] = df[label_column]

    # Stratified 80/20 split to preserve class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label_numeric"],
        test_size=0.20,
        random_state=42,
        stratify=df["label_numeric"],
    )

    print(f"\nTraining set: {len(X_train)} reviews")
    print(f"Test set:     {len(X_test)} reviews")

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
        sublinear_tf=True,
    )

    # Fit only on training data to avoid leakage.
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")

    feature_names = tfidf.get_feature_names_out()
    print(f"Example features (first 20): {list(feature_names[:20])}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf, label_map


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train four classifiers and collect test metrics plus 5-fold CV scores."""
    print("\n" + "=" * 60)
    print("Step 5: Model training and evaluation")
    print("=" * 60)

    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # 5-fold CV on the training set for a generalisation estimate.
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
        }

        print(f"  Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV Score:  {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
        print("\n  Classification report:")
        print(classification_report(y_test, y_pred, target_names=["Real (OR)", "Fake (CG)"]))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, y_test):
    """Generate confusion matrices, metric comparison, and CV summary plots."""
    print("\n" + "=" * 60)
    print("Step 6: Plotting")
    print("=" * 60)

    model_names = list(results.keys())

    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_test, data["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real (OR)", "Fake (CG)"],
            yticklabels=["Real (OR)", "Fake (CG)"],
            ax=ax,
        )
        ax.set_title(f"{name}\n(Accuracy: {data['accuracy']:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("03_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 03_confusion_matrices.png")

    # Metric comparison
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[name][metric] for name in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_model_karsilastirma.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 04_model_karsilastirma.png")

    # Cross-validation summary
    fig, ax = plt.subplots(figsize=(10, 6))
    cv_means = [results[name]["cv_mean"] for name in model_names]
    cv_stds = [results[name]["cv_std"] for name in model_names]

    bars = ax.bar(
        model_names,
        cv_means,
        yerr=cv_stds,
        capsize=5,
        color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.4f}\n+/-{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold Cross-Validation Results", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("05_cross_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved 05_cross_validation.png")


def generate_summary_table(results):
    """Build a summary DataFrame and write it to CSV."""
    print("\n" + "=" * 60)
    print("Summary table")
    print("=" * 60)

    summary = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [f"{results[n]['accuracy']:.4f}" for n in results],
        "Precision": [f"{results[n]['precision']:.4f}" for n in results],
        "Recall": [f"{results[n]['recall']:.4f}" for n in results],
        "F1 Score": [f"{results[n]['f1_score']:.4f}" for n in results],
        "CV Mean": [f"{results[n]['cv_mean']:.4f}" for n in results],
        "CV Std": [f"+/-{results[n]['cv_std']:.4f}" for n in results],
    })

    print(summary.to_string(index=False))
    summary.to_csv("06_sonuc_tablosu.csv", index=False)
    print("\nSaved 06_sonuc_tablosu.csv")

    best_model = max(results, key=lambda x: results[x]["f1_score"])
    print(f"\nBest model: {best_model}")
    print(f"  F1 Score: {results[best_model]['f1_score']:.4f}")
    print(f"  Accuracy: {results[best_model]['accuracy']:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # Update these to match your local file and column names if needed.
    DATA_FILE = "fake_reviews_dataset.csv"
    TEXT_COLUMN = "text_"
    LABEL_COLUMN = "label"

    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        print("Download the dataset and place the CSV in the project root.")
        print("https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset")
        return

    print("Starting fake review detection pipeline...\n")

    df = load_data(DATA_FILE)
    df = explore_data(df, TEXT_COLUMN, LABEL_COLUMN)
    df = preprocess_dataframe(df, TEXT_COLUMN, LABEL_COLUMN)

    X_train, X_test, y_train, y_test, tfidf, label_map = extract_features(
        df, LABEL_COLUMN, max_features=10000
    )

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    plot_results(results, y_test)
    generate_summary_table(results)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print("\nGenerated files:")
    print("  01_eda_analizi.png         - EDA plots")
    print("  02_kelime_bulutu.png       - Word clouds")
    print("  03_confusion_matrices.png  - Confusion matrices")
    print("  04_model_karsilastirma.png - Metric comparison")
    print("  05_cross_validation.png    - CV results")
    print("  06_sonuc_tablosu.csv       - Results summary")


if __name__ == "__main__":
    main()
