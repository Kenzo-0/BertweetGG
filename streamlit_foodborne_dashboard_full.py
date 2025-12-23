import streamlit as st
import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit.components.v1 as components

# ------------------ Config ------------------
st.set_page_config(page_title="Foodborne Illness Detector", page_icon="🍲", layout="wide")

# ------------------ Model Paths (Hugging Face public repos) ------------------
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
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, output_attentions=True)
    model.eval()
    return tokenizer, model

# ------------------ Prediction ------------------
def predict_text(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=1).numpy()[0]
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

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1)
        return probs.cpu().numpy()

    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > max_tokens:
        text_to_explain = tokenizer.convert_tokens_to_string(tokenized_text[:max_tokens])
    else:
        text_to_explain = text

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(model_predict, masker, output_names=LABELS)
    shap_values = explainer([text_to_explain])
    return shap_values

# ------------------ Streamlit SHAP renderer ------------------
def st_shap(plot_html, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot_html}</body>"
    components.html(shap_html, height=height or 300)

# ------------------ Neon SHAP Text ------------------
def shap_text_neon(shap_values, color_positive="#39FF14", color_negative="#FF6EC7"):
    text = ""
    for token, val in zip(shap_values.data, shap_values.values[0]):
        if not token.strip():  # skip empty tokens
            continue
        color = color_positive if val > 0 else color_negative
        opacity = min(abs(val) * 5, 1)
        text += f'<span style="color:{color}; opacity:{opacity}; font-weight:bold">{token} </span>'
    return text

# ------------------ Plain-language SHAP explanation ------------------
def shap_plain_text_user(shap_values, predicted_label, label_names):
    if len(shap_values.data) == 0 or shap_values.values.size == 0:
        return "No significant words found."

    class_idx = label_names.index(predicted_label)

    if shap_values.values.ndim == 3:
        token_values = shap_values.values[0, :, class_idx]
    else:
        token_values = shap_values.values[0]

    tokens_and_values = [(t, v) for t, v in zip(shap_values.data, token_values) if t.strip()]
    top_tokens = sorted(tokens_and_values, key=lambda x: abs(x[1]), reverse=True)[:5]

    explanation = []
    for token, val in top_tokens:
        impact = "increases" if val > 0 else "decreases"
        explanation.append(f'"{token}" {impact} likelihood of {predicted_label}')

    if not explanation:
        return "No words had significant impact on prediction."

    return " | ".join(explanation)

# ------------------ Attention Keywords ------------------
def get_attention_keywords(outputs, inputs, tokenizer, top_k=6):
    attns = outputs.attentions[-1]
    scores = attns.mean(dim=1).mean(dim=1).squeeze()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    token_scores = [(tok, s) for tok, s in zip(tokens, scores.tolist()) if tok not in tokenizer.all_special_tokens]
    return sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_k]

# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["📘 About", "🔍 Text Analysis"])

# About
with tab1:
    st.header("About Foodborne Illness Detection")
    st.markdown("""
This dashboard uses **BERTweet models** to classify foodborne illness risk from text reviews.

- **SHAP** highlights token contribution (neon/heatmap colors + plain language explanation)
- **Attention** highlights important keywords
- Compare multiple trained models
""")
    st.header("Symptoms of Foodborne Illness")
    st.image("https://www.cfs.gov.hk/english/trade_zone/safe_kitchen/image/p16-1.png", width=600)

# Analysis
with tab2:
    st.header("Text Risk Analysis")
    model_choice = st.selectbox("Select BERTweet model", list(MODEL_PATHS.keys()))
    text_input = st.text_area("Paste a review / complaint:", height=160)

    if st.button("Analyze Risk") and text_input.strip():
        with st.spinner("Analyzing..."):
            tokenizer, model = load_model(MODEL_PATHS[model_choice])
            label, probs, outputs, enc = predict_text(text_input, tokenizer, model)

        # ---------------- Prediction ----------------
        st.subheader("🧠 Prediction")
        st.success(f"Prediction: **{label}**")

        # ---------------- SHAP Plain Language & Neon Highlights ----------------
        shap_values = get_shap_explanation(text_input, tokenizer, model, max_tokens=50)

        st.subheader("📝 SHAP Explanation (Plain Language)")
        plain_text = shap_plain_text_user(shap_values[0], label, LABELS)
        st.markdown(plain_text)

        # ---------------- Class Probabilities ----------------
        st.subheader("📊 Class Probabilities")
        st.bar_chart(probs)

        # ---------------- Attention Keywords ----------------
        st.subheader("🔑 Attention Keywords")
        keywords = get_attention_keywords(outputs, enc, tokenizer)
        for tok, score in keywords:
            st.markdown(f"- **{tok}** (attention = {score:.4f})")

        # ---------------- Optional SHAP Graph ----------------
        st.subheader("📈 SHAP Graph (Optional)")
        shap_plot_html = shap.plots.text(shap_values[0], display=False)
        st_shap(shap_plot_html, height=200)

    else:
        st.info("Enter text and click Analyze Risk")
