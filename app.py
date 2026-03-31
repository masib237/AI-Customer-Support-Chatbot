import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="AI Customer Support", page_icon="🤖")
st.title("🤖 AI Customer Support Chatbot")

# 1. Load model and tokenizer directly (Avoids the Pipeline task error)
@st.cache_resource
def load_model():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# STEP 3: KNOWLEDGE BASE
# =========================
knowledge_base = {
    "General Inquiry": {
        "what services do you offer": "We offer education abroad support, visa assistance, and investment advisory services.",
        "how can i contact you": "You can contact us via email, phone, or our website chat support.",
        "where are you located": "We are based in Kenya and serve both local and international clients."
    },

    "Education Abroad": {
        "requirements for studying abroad": "You need academic transcripts, passport, application forms, and proof of funds.",
        "best countries for kenyan students": "Popular destinations include Canada, UK, USA, Germany, and Australia.",
        "how to apply for scholarships": "You apply through university websites or scholarship portals like DAAD or Chevening."
    },

    "Visa Assistance": {
        "documents for visa application": "You typically need a passport, application form, bank statements, and invitation letter if applicable.",
        "how long does visa take": "Processing time varies: 2–6 weeks depending on the country.",
        "visa rejection reasons": "Common reasons include incomplete documents, weak financial proof, or unclear travel purpose."
    },

    "Investment & Financial Services": {
        "diaspora investment options": "You can invest in real estate, savings accounts, and government bonds.",
        "how to invest from abroad": "You can invest through diaspora banking services or licensed financial advisors.",
        "safe investment options": "Low-risk options include fixed deposits, treasury bonds, and SACCOs."
    }
}

if "messages" not in st.session_state:
    st.session_state.messages = []

category = st.selectbox(
    "📂 What would you like help with?",
    ["General Inquiry", "Education Abroad", "Visa Assistance", "Investment & Financial Services"]
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question...")

# =========================
# STEP 4: RETRIEVAL FUNCTION
# =========================
def get_context(category, user_input):
    user_input = user_input.lower()

    if category not in knowledge_base:
        return None

    best_match = None

    for question, answer in knowledge_base[category].items():
        if question in user_input:
            best_match = answer
            break

    return best_match

# =========================
# STEP 5: UPDATED PROMPT LOGIC
# =========================
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = get_context(category, user_input)

    if context:
        prompt = f"""
You are a professional customer support assistant.

Use the context below to answer the user.

Context:
{context}

User question:
{user_input}

Give a clear, helpful response.
"""
    else:
        prompt = f"""
You are a customer support assistant.

The user asked:
{user_input}

If you don't know the exact answer, respond politely and say you will escalate the query.
"""

    with st.chat_message("assistant"):
        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not answer.strip():
            answer = "I couldn't find a clear answer. Please rephrase your question."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})