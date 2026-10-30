# Research Article Classification — Streamlit App

## Project structure
```text
research_article_classifier_ready/
├── app.py
├── requirements.txt
├── README.md
├── sample_input.csv
└── artifacts/
    └── best_model.joblib
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly
1. Extract this ZIP.
2. Upload all files and folders (including `artifacts/best_model.joblib`) to a GitHub repository.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Create an app from that repository, branch `main`, main file `app.py`.
5. Deploy and share the generated `*.streamlit.app` URL.

Free Streamlit Community Cloud apps may hibernate after inactivity; they are not guaranteed to run continuously 24/7.

## CSV input
Upload a CSV with a column containing abstracts. The app suggests common column names
such as `abstract`, `text`, and `content`, or lets the visitor select another column.
The output includes `predicted_label` and `prediction_status`, and can be downloaded.

## Model compatibility and safety
The model is copied from the supplied `best_model.joblib`. Serialized scikit-learn
models can be version-sensitive. If deployment reports a version/import error, use
the scikit-learn version from the training environment and update requirements.txt.
Only load joblib files from trusted sources because deserialization can execute code.
