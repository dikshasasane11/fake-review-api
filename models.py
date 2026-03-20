"""
FakeReviewDetector
==================
Encapsulates all ML logic: feature extraction, training, and prediction.
Designed to be instantiated once at API startup and reused for all requests.
"""

import re
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


class FakeReviewDetector:

    TFIDF_FEATURES = 5_000
    TEST_SIZE      = 0.20
    RANDOM_SEED    = 42

    def __init__(self):
        self.tfidf      : TfidfVectorizer   = None
        self.model      : LogisticRegression = None
        self.is_trained : bool               = False
        self._meta      : dict               = {}

    # ── Feature Engineering ───────────────────────────────────────────────────

    @staticmethod
    def _handcrafted(text: str) -> dict:
        words     = text.split()
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        return {
            "word_count"        : len(words),
            "char_count"        : len(text),
            "avg_word_len"      : float(np.mean([len(w) for w in words])) if words else 0.0,
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

    @staticmethod
    def _label_fake(row: pd.Series) -> int:
        text  = row["Review"]
        words = text.split()
        wc    = len(words)
        score = 0
        if wc < 20:                                             score += 2
        if text.count("!") > 5:                                score += 1
        if len(set(words)) / max(wc, 1) < 0.5:                score += 1
        if row["Rating"] in (1, 5) and wc < 30:               score += 1
        return int(score >= 2)

    def _build_X(self, texts, feat_df, fit: bool = False):
        if fit:
            tfidf_mat = self.tfidf.fit_transform(texts)
        else:
            tfidf_mat = self.tfidf.transform(texts)
        return hstack([tfidf_mat, csr_matrix(feat_df.values)])

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path).dropna()
        df["is_fake"] = df.apply(self._label_fake, axis=1)

        feat_df = df["Review"].apply(self._handcrafted).apply(pd.Series)

        X_tr_t, X_te_t, X_tr_f, X_te_f, y_train, y_test = train_test_split(
            df["Review"], feat_df, df["is_fake"],
            test_size    = self.TEST_SIZE,
            random_state = self.RANDOM_SEED,
            stratify     = df["is_fake"],
        )

        self.tfidf = TfidfVectorizer(
            max_features = self.TFIDF_FEATURES,
            ngram_range  = (1, 2),
            sublinear_tf = True,
        )

        X_train = self._build_X(X_tr_t, X_tr_f, fit=True)
        X_test  = self._build_X(X_te_t, X_te_f, fit=False)

        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rep    = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cls    = rep.get("1", rep.get("weighted avg", {}))

        # Top features
        feat_names = self.tfidf.get_feature_names_out().tolist() + list(feat_df.columns)
        coefs      = self.model.coef_[0]
        top_fake   = [[feat_names[i], round(float(coefs[i]), 4)]
                      for i in np.argsort(coefs)[-10:][::-1]]
        top_real   = [[feat_names[i], round(float(coefs[i]), 4)]
                      for i in np.argsort(coefs)[:10]]

        self._meta = {
            "model_name"            : "Fake Review Detector v1.0",
            "algorithm"             : "Logistic Regression + TF-IDF (bigrams)",
            "tfidf_features"        : self.TFIDF_FEATURES,
            "handcrafted_features"  : len(feat_df.columns),
            "total_features"        : self.TFIDF_FEATURES + len(feat_df.columns),
            "training_samples"      : len(X_tr_t),
            "test_accuracy"         : round(accuracy_score(y_test, y_pred), 4),
            "test_precision"        : round(cls.get("precision", 0), 4),
            "test_recall"           : round(cls.get("recall",    0), 4),
            "test_f1"               : round(cls.get("f1-score",  0), 4),
            "top_fake_features"     : top_fake,
            "top_real_features"     : top_real,
        }
        self.is_trained = True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, text: str, rating: int = 4) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call .train() first.")

        feat    = pd.DataFrame([self._handcrafted(text)])
        tfidf_v = self.tfidf.transform([text])
        X       = hstack([tfidf_v, csr_matrix(feat.values)])

        pred = int(self.model.predict(X)[0])
        prob = float(self.model.predict_proba(X)[0][pred])

        return {
            "label"     : "FAKE" if pred == 1 else "GENUINE",
            "is_fake"   : bool(pred),
            "confidence": round(prob, 4),
            "signals"   : {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in self._handcrafted(text).items()},
        }

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return self._meta
