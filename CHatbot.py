import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
import base64
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

# --- Set background image ---
def set_bg(image_file):
    with open(image_file, "rb") as img:
        img_bytes = img.read()
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
            width: 100vw;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Error handling wrapper ---
def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {e}")
            st.write(e)
    return wrapper

def load_and_create_db():
    pdf_dir = "pdfs"  # Use relative path for deployment

    docs = []
    if os.path.isdir(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embeddings)
        return vectordb
    else:
        raise ValueError("PDF directory does not exist.")

@safe_run
def main():
    st.title("RadioBot")
    set_bg("Thom.png")

    vectordb = load_and_create_db()

    chat_llm = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.1,
        max_tokens=512,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=vectordb.as_retriever(),
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    # Display chat history
    for user_msg, bot_msg in st.session_state["history"]:
        st.markdown(
            f"""
            <div style="background-color: rgba(240,240,240,0.85); padding: 0.5em 1em; border-radius: 8px; margin-bottom: 0.5em;">
                <b>You:</b> {user_msg}
            </div>
            <div style="background-color: rgba(255,255,255,0.95); padding: 0.5em 1em; border-radius: 8px; margin-bottom: 1em;">
                <b>RadioBot:</b> {bot_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    user_input = st.text_input("Your question:", key="user_input")

    if st.button("Send"):
        if user_input:
            # Use the actual history for follow-ups, or [] for stateless
            chat_history = [
                {"role": "user", "content": q}
                if i % 2 == 0 else
                {"role": "assistant", "content": a}
                for i, (q, a) in enumerate(st.session_state["history"])
            ]
            response = chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })
            st.session_state["history"].append((user_input, response["answer"]))
            # Do NOT try to clear st.session_state["user_input"] here

if __name__ == "__main__":
    main()