"""
Fake Review Detection System
==============================
Dataset : TripAdvisor Hotel Reviews (20,491 reviews)
Features: TF-IDF (5,000 n-grams) + 9 handcrafted text signals
Models  : Logistic Regression · Random Forest · Linear SVM · Naive Bayes

Usage
-----
    python fake_review_detector.py                  # train & evaluate
    python fake_review_detector.py --predict "..."  # predict single review
"""

import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── 1. Configuration ──────────────────────────────────────────────────────────

DATA_PATH   = "tripadvisor_hotel_reviews.csv"
TFIDF_FEATS = 5_000
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ── 2. Feature Engineering ───────────────────────────────────────────────────

def extract_handcrafted_features(text: str) -> dict:
    """
    Nine lightweight signals that correlate with review authenticity:
      - word_count            : very short reviews are suspicious
      - char_count            : raw length
      - avg_word_len          : unusual vocabulary density
      - exclamation_count     : overuse of '!' is a spam signal
      - unique_word_ratio     : low ratio → copy-paste or bot
      - sentence_count        : fragmented text → scripted
      - avg_sentence_len      : abnormally short sentences
      - pronoun_count         : I/me/my/we saturation
      - superlative_count     : excessive positive/negative extremes
    """
    words     = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    return {
        "word_count"        : len(words),
        "char_count"        : len(text),
        "avg_word_len"      : np.mean([len(w) for w in words]) if words else 0,
        "exclamation_count" : text.count("!"),
        "unique_word_ratio" : len(set(words)) / max(len(words), 1),
        "sentence_count"    : len(sentences),
        "avg_sentence_len"  : len(words) / max(len(sentences), 1),
        "pronoun_count"     : sum(1 for w in words
                                  if w.lower() in {"i","me","my","we","our","us"}),
        "superlative_count" : sum(1 for w in words
                                  if w.lower() in {
                                      "best","worst","amazing","terrible",
                                      "perfect","awful","excellent","horrible"
                                  }),
    }


def label_fake(row: pd.Series) -> int:
    """
    Rule-based heuristic label (used as ground truth for training).
    A review is flagged as fake when at least 2 of the following hold:
      · fewer than 20 words  (score +2)
      · more than 5 '!'      (score +1)
      · unique-word ratio < 0.5 (score +1)
      · extreme rating (1 or 5) + fewer than 30 words (score +1)
    """
    text  = row["Review"]
    words = text.split()
    wc    = len(words)
    score = 0
    if wc < 20:                                                score += 2
    if text.count("!") > 5:                                    score += 1
    if len(set(words)) / max(wc, 1) < 0.5:                    score += 1
    if row["Rating"] in (1, 5) and wc < 30:                   score += 1
    return int(score >= 2)


# ── 3. Data Loading & Preprocessing ──────────────────────────────────────────

def load_data(path: str):
    df = pd.read_csv(path).dropna()
    df["is_fake"] = df.apply(label_fake, axis=1)

    feat_df = df["Review"].apply(extract_handcrafted_features).apply(pd.Series)

    print(f"\n{'─'*55}")
    print(f"  Dataset  : {len(df):,} reviews")
    print(f"  Genuine  : {(df['is_fake']==0).sum():,}  "
          f"({(df['is_fake']==0).mean()*100:.1f}%)")
    print(f"  Fake     : {df['is_fake'].sum():,}  "
          f"({df['is_fake'].mean()*100:.1f}%)")
    print(f"{'─'*55}\n")

    return df, feat_df


# ── 4. Build Combined Feature Matrix ─────────────────────────────────────────

def build_features(df, feat_df):
    """Combine TF-IDF text features with handcrafted numeric features."""
    tfidf = TfidfVectorizer(
        max_features  = TFIDF_FEATS,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
    )

    X_text = df["Review"]
    X_feat = feat_df
    y      = df["is_fake"]

    (X_train_t, X_test_t,
     X_train_f, X_test_f,
     y_train,   y_test) = train_test_split(
        X_text, X_feat, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y,
    )

    X_train_tfidf = tfidf.fit_transform(X_train_t)
    X_test_tfidf  = tfidf.transform(X_test_t)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_f.values)])
    X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_f.values)])

    return X_train, X_test, y_train, y_test, tfidf, feat_df.columns.tolist()


# ── 5. Model Training & Evaluation ───────────────────────────────────────────

MODELS = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, C=1.0),
    "Random Forest"       : RandomForestClassifier(n_estimators=100,
                                                    random_state=RANDOM_SEED),
    "Linear SVM"          : LinearSVC(max_iter=2000),
    "Naive Bayes"         : MultinomialNB(),
}


def train_and_evaluate(X_train, X_test, y_train, y_test):
    trained = {}
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8}")
    print("─" * 65)

    for name, model in MODELS.items():
        Xtr = abs(X_train) if name == "Naive Bayes" else X_train
        Xte = abs(X_test)  if name == "Naive Bayes" else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cls = rep.get("1", rep.get("weighted avg", {}))

        acc  = accuracy_score(y_test, y_pred)
        prec = cls.get("precision", 0)
        rec  = cls.get("recall",    0)
        f1   = cls.get("f1-score",  0)

        print(f"{name:<25} {acc*100:>8.2f}%  {prec*100:>8.2f}%  "
              f"{rec*100:>7.2f}%  {f1*100:>7.2f}%")

        trained[name] = model

    print()
    return trained


# ── 6. Feature Importance ─────────────────────────────────────────────────────

def print_feature_importance(lr_model, tfidf, handcrafted_cols, top_n=10):
    feature_names = tfidf.get_feature_names_out().tolist() + handcrafted_cols
    coefs         = lr_model.coef_[0]

    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    print(f"\n{'─'*55}")
    print("  Top features → FAKE reviews (positive coefficient)")
    print(f"{'─'*55}")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:<30}  {coefs[idx]:+.4f}")

    print(f"\n{'─'*55}")
    print("  Top features → GENUINE reviews (negative coefficient)")
    print(f"{'─'*55}")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:<30}  {coefs[idx]:+.4f}")
    print()


# ── 7. Live Prediction Helper ─────────────────────────────────────────────────

def predict_review(text: str, model, tfidf, rating: int = 4) -> dict:
    """
    Predict whether a single review is fake.

    Parameters
    ----------
    text   : review text
    model  : trained classifier
    tfidf  : fitted TfidfVectorizer
    rating : 1-5 star rating (default 4)

    Returns
    -------
    dict with keys: label, confidence, signals
    """
    row       = pd.Series({"Review": text, "Rating": rating})
    feat      = pd.DataFrame([extract_handcrafted_features(text)])
    tfidf_vec = tfidf.transform([text])
    X         = hstack([tfidf_vec, csr_matrix(feat.values)])

    if isinstance(model, MultinomialNB):
        X = abs(X)

    pred = model.predict(X)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(model.predict_proba(X)[0][pred]), 3)
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        confidence = round(1 / (1 + np.exp(-abs(score))), 3)

    return {
        "label"      : "FAKE" if pred == 1 else "GENUINE",
        "confidence" : confidence,
        "signals"    : {k: round(v, 3) for k, v in
                        extract_handcrafted_features(text).items()},
    }


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fake Review Detection System")
    parser.add_argument("--data",    default=DATA_PATH,
                        help="Path to CSV dataset")
    parser.add_argument("--predict", default=None,
                        help="Predict a single review text")
    parser.add_argument("--rating",  type=int, default=4,
                        help="Star rating for the --predict review (1-5)")
    args = parser.parse_args()

    # Load & preprocess
    df, feat_df = load_data(args.data)

    # Build features
    X_train, X_test, y_train, y_test, tfidf, hc_cols = build_features(df, feat_df)

    # Train all models
    print("Training models …\n")
    trained = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Best model = Logistic Regression
    best_model = trained["Logistic Regression"]

    # Feature importance
    print_feature_importance(best_model, tfidf, hc_cols)

    # Confusion matrix for best model
    Xte      = X_test
    y_pred   = best_model.predict(Xte)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix (Logistic Regression)")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}\n")

    # Optional: predict a custom review
    if args.predict:
        result = predict_review(args.predict, best_model, tfidf, args.rating)
        print(f"\n{'─'*55}")
        print(f"  Input   : {args.predict[:80]}")
        print(f"  Label   : {result['label']}")
        print(f"  Conf.   : {result['confidence']}")
        print(f"  Signals : {json.dumps(result['signals'], indent=4)}")
        print(f"{'─'*55}\n")
        return

    # Demo predictions on sample reviews
    samples = [
        ("AMAZING!!! BEST HOTEL EVER!!! PERFECT STAY!!!",          5),
        ("Great location and friendly staff. Room was clean. "
         "Minor noise from street at night but overall a "
         "pleasant experience. Would recommend.",                   4),
        ("WORST STAY EVER! DO NOT BOOK! TERRIBLE! AVOID!!!",       1),
        ("Stayed for 3 nights. Breakfast was decent, room service "
         "prompt, AC worked well. Bed slightly firm. Good value.",  4),
    ]

    print(f"\n{'─'*55}")
    print("  Sample Predictions")
    print(f"{'─'*55}")
    for text, rating in samples:
        res = predict_review(text, best_model, tfidf, rating)
        tag = "🚩 FAKE   " if res["label"] == "FAKE" else "✅ GENUINE"
        print(f"  [{tag}]  {text[:60]}")
    print()


if __name__ == "__main__":
    main()
