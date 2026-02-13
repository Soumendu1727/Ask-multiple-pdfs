import streamlit as st
import os
import faiss
import numpy as np
import torch
import json

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# CONFIG
# -----------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_PATH = "./models/phi-2"
CHAT_HISTORY_FILE = "chat_history.json"

# -----------------------------
# LOAD MODELS (Load Once)
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float32
    )

    model.to("cpu")   # force everything to CPU
    model.eval()

    return tokenizer, model

embedding_model = load_embedding_model()
tokenizer, llm_model = load_llm()

# -----------------------------
# CHAT HISTORY STORAGE
# -----------------------------

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------
# TEXT CHUNKING
# -----------------------------

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# -----------------------------
# EMBEDDINGS + FAISS
# -----------------------------

def get_embeddings(text_chunks):
    embeddings = embedding_model.encode(
        text_chunks,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -----------------------------
# RETRIEVAL
# -----------------------------

def retrieve(query, index, text_chunks, k=3):
    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_vector, k)
    return [text_chunks[i] for i in indices[0]]

# -----------------------------
# LOCAL LLM
# -----------------------------

def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an intelligent assistant.
Answer strictly using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    print("Prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    return response.strip()

# -----------------------------
# STREAMLIT APP
# -----------------------------

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.title("📚 Chat with Multiple PDFs (Local RAG + Persistent Memory)")

    # Initialize session state
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.text_chunks = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Display previous chat
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # User input
    user_question = st.chat_input("Ask a question about your documents")

    if user_question and st.session_state.index is not None:

        # 1️⃣ Show user message instantly
        with st.chat_message("user"):
            st.write(user_question)

        # 2️⃣ Create assistant message container
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown("⏳ Thinking...")

            # 3️⃣ Generate answer
            relevant_chunks = retrieve(
                user_question,
                st.session_state.index,
                st.session_state.text_chunks
            )

            answer = ask_llm(user_question, relevant_chunks)

            # 4️⃣ Replace loading with real answer
            answer_placeholder.markdown(answer)

        # 5️⃣ Save chat history AFTER generation
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

        save_chat_history(st.session_state.chat_history)

    # Sidebar
    with st.sidebar:
        st.subheader("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True
        )

        if st.button("Process") and pdf_docs:
            with st.spinner("Processing documents..."):
                raw_text = extract_text_from_pdfs(pdf_docs)
                text_chunks = split_text(raw_text)

                embeddings = get_embeddings(text_chunks)
                index = create_faiss_index(embeddings)

                st.session_state.index = index
                st.session_state.text_chunks = text_chunks

            st.success("Documents processed successfully!")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.success("Chat history cleared!")
            st.rerun()

# -----------------------------

if __name__ == "__main__":
    main()
