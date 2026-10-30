from pathlib import Path
import re
import sys

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# The saved notebook pipeline refers to __main__.NLPPreprocessor.
# Define the same class before joblib.load() so that the pickle can resolve it.
class NLPPreprocessor(BaseEstimator, TransformerMixin):
    """Explicit NLP: cleaning, tokenization, stop-word removal, lemmatization."""
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess(text) for text in X]

    def preprocess(self, text):
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        return " ".join(tokens)

# Ensure the custom class is visible under the exact module/name stored in the pickle.
sys.modules["__main__"].NLPPreprocessor = NLPPreprocessor

st.set_page_config(page_title="Research Article Classification", page_icon="📚", layout="wide")
MODEL_PATH = Path(__file__).parent / "artifacts" / "best_model.joblib"

@st.cache_resource
def load_model():
    # The notebook's NLPPreprocessor uses these NLTK corpora at prediction time.
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return joblib.load(MODEL_PATH)

st.title("📚 Research Article Classification")
st.write("Upload research abstracts and classify them into **Science**, **Business**, or **Social Science**.")
st.caption("Machine Learning project | Text classification using your trained model")

if not MODEL_PATH.exists():
    st.error("Model file missing: artifacts/best_model.joblib")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error("The model could not be loaded. Check the Python and scikit-learn versions and the model file.")
    st.exception(exc)
    st.stop()

tab_csv, tab_single, tab_about = st.tabs(["📁 Upload CSV", "✍️ Single abstract", "ℹ️ About"])

with tab_csv:
    st.subheader("Classify a CSV dataset")
    st.write("Upload a CSV containing one research abstract per row.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            try:
                df = pd.read_csv(uploaded)
            except UnicodeDecodeError:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding="latin-1")
            if df.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                st.write("Dataset preview")
                st.dataframe(df.head(10), use_container_width=True)
                likely = {"abstract", "abstracts", "text", "content", "research_abstract", "paper_abstract"}
                matches = [c for c in df.columns if str(c).strip().lower() in likely]
                default_index = list(df.columns).index(matches[0]) if matches else 0
                col = st.selectbox("Select the abstract column", list(df.columns), index=default_index)
                if st.button("Predict categories", type="primary"):
                    texts = df[col].fillna("").astype(str).str.strip()
                    valid = texts.ne("")
                    if not valid.any():
                        st.warning("No non-empty abstracts found.")
                    else:
                        with st.spinner("Classifying abstracts..."):
                            result = df.copy()
                            preds = pd.Series([""] * len(df), index=df.index, dtype="object")
                            preds.loc[valid] = model.predict(texts.loc[valid].tolist())
                            result["predicted_label"] = preds
                            result["prediction_status"] = valid.map({True: "Classified", False: "Empty abstract"})
                        st.success(f"Predicted labels for {int(valid.sum()):,} abstracts.")
                        summary = result.loc[valid, "predicted_label"].value_counts().rename_axis("Predicted category").reset_index(name="Number of abstracts")
                        left, right = st.columns(2)
                        with left:
                            st.subheader("Prediction summary")
                            st.dataframe(summary, use_container_width=True)
                        with right:
                            st.subheader("Category distribution")
                            st.bar_chart(summary.set_index("Predicted category")["Number of abstracts"])
                        st.subheader("Classified dataset")
                        st.dataframe(result, use_container_width=True)
                        st.download_button(
                            "⬇️ Download results as CSV",
                            result.to_csv(index=False).encode("utf-8-sig"),
                            "research_abstracts_predictions.csv",
                            "text/csv",
                            type="primary",
                        )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

with tab_single:
    st.subheader("Predict one research abstract")
    abstract = st.text_area("Paste the abstract", height=220, placeholder="Paste a research paper abstract here...")
    if st.button("Predict category", type="primary", key="single_prediction"):
        if not abstract.strip():
            st.warning("Enter an abstract first.")
        else:
            try:
                label = model.predict([abstract.strip()])[0]
                st.success(f"Predicted category: **{label}**")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_about:
    st.subheader("About this application")
    st.markdown("""
    This application uses your trained machine-learning pipeline to classify
    research abstracts into **Science**, **Business**, or **Social Science**.
    Predictions are automated estimates and should be reviewed when academic
    accuracy is important.
    """)

st.divider()
st.caption("Research Article Classification | Academic project")
