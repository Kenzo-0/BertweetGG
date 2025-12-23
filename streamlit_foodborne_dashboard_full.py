import streamlit as st
import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# ------------------ Config ------------------
st.set_page_config(
    page_title="Foodborne Illness Detector",
    page_icon="🍲",
    layout="wide"
)

# ------------------ Model Paths ------------------
MODEL_PATHS = {
    "BERTweet (Baseline)": "Kenzo15/bertweet-smote",
    "BERTweet + SMOTE": "Kenzo15/bertweet-baseline",
    "BERTweet (Hyper-tuned)": "Kenzo15/final-bertweet",
}

LABELS = ["Low Risk", "High Risk"]

# ------------------ Sidebar ------------------
st.sidebar.title("Foodborne Illness Detector")
st.sidebar.markdown("---")
st.sidebar.caption("Created by Iskandar")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        output_attentions=True
    )
    model.eval()
    return tokenizer, model

# ------------------ Prediction ------------------
def predict_text(text, tokenizer, model):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        out = model(**enc)

    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    label = LABELS[int(np.argmax(probs))]

    return label, probs, out, enc

# ------------------ SHAP Explanation ------------------
def get_shap_explanation(text, tokenizer, model, max_tokens=50):

    def model_predict(texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            raise ValueError("Input must be str or list of str")

        texts = [str(t) for t in texts]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1)

        return probs.cpu().numpy()

    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        text = tokenizer.convert_tokens_to_string(tokens[:max_tokens])

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        model_predict,
        masker,
        output_names=LABELS
    )

    shap_values = explainer([text])
    return shap_values

# ------------------ Plain-language SHAP ------------------
def shap_plain_text_user(shap_values, predicted_label, label_names):
    class_idx = label_names.index(predicted_label)

    if shap_values.values.ndim == 3:
        token_values = shap_values.values[0, :, class_idx]
    else:
        token_values = shap_values.values[0]

    tokens = shap_values.data
    pairs = [(t, v) for t, v in zip(tokens, token_values) if t.strip()]
    top = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

    if not top:
        return "No words had significant impact on the prediction."

    explanations = []
    for token, val in top:
        impact = "increases" if val > 0 else "decreases"
        explanations.append(
            f'"{token}" {impact} likelihood of {predicted_label}'
        )

    return " | ".join(explanations)

# ------------------ SHAP Bar Chart ------------------
def shap_bar_data(shap_values, predicted_label, label_names, top_k=8):
    class_idx = label_names.index(predicted_label)

    if shap_values.values.ndim == 3:
        values = shap_values.values[0, :, class_idx]
    else:
        values = shap_values.values[0]

    tokens = shap_values.data
    pairs = [(t, v) for t, v in zip(tokens, values) if t.strip()]
    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    df = pd.DataFrame(pairs, columns=["Token", "SHAP Value"])
    return df

# ------------------ Attention Keywords ------------------
def get_attention_keywords(outputs, inputs, tokenizer, top_k=6):
    attn = outputs.attentions[-1]
    scores = attn.mean(dim=1).mean(dim=1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze()
    )

    pairs = [
        (tok, score)
        for tok, score in zip(tokens, scores.tolist())
        if tok not in tokenizer.all_special_tokens
    ]

    return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]

# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["📘 About", "🔍 Text Analysis"])

# ------------------ About ------------------
with tab1:
    st.header("About Foodborne Illness Detection")
    st.markdown("""
This system uses natural language processing (NLP) and BERTweet-based deep learning models 
to analyze food-related reviews and complaints. By examining textual patterns associated 
with symptoms, hygiene issues, and contamination indicators, the model predicts the 
potential risk of foodborne illness. Explainable AI techniques such as SHAP and attention 
mechanisms are applied to provide transparent and interpretable predictions.
    """)

    st.header("Symptoms of Foodborne Illness")
    st.image(
        "https://www.cfs.gov.hk/english/trade_zone/safe_kitchen/image/p16-1.png",
        width=600
    )

# ------------------ Analysis ------------------
with tab2:
    st.header("Text Risk Analysis")

    model_choice = st.selectbox(
        "Select BERTweet model",
        list(MODEL_PATHS.keys())
    )

    text_input = st.text_area(
        "Paste a review / complaint:",
        height=160
    )

    if st.button("Analyze Risk") and text_input.strip():

        with st.spinner("Analyzing..."):
            tokenizer, model = load_model(MODEL_PATHS[model_choice])
            label, probs, outputs, enc = predict_text(
                text_input, tokenizer, model
            )

        # Prediction
        st.subheader("🧠 Prediction")
        st.success(f"Prediction: **{label}**")

        # SHAP explanation
        shap_values = get_shap_explanation(
            text_input, tokenizer, model
        )

        st.subheader("📝 SHAP Explanation")
        st.markdown(
            shap_plain_text_user(
                shap_values[0],
                label,
                LABELS
            )
        )

        # ---------------- SHAP Bar Chart ----------------
        st.subheader("📊 SHAP Token Contributions (Bar Chart)")
        bar_df = shap_bar_data(shap_values[0], label, LABELS)
        st.bar_chart(bar_df.set_index("Token"))

        # Probabilities
        st.subheader("📊 Class Probabilities")
        st.bar_chart(probs)

        # Attention
        st.subheader("🔑 Attention Keywords")
        for tok, score in get_attention_keywords(
            outputs, enc, tokenizer
        ):
            st.markdown(f"- **{tok}** (attention = {score:.4f})")

    else:
        st.info("Enter text and click Analyze Risk")
