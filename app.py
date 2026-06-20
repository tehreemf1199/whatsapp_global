from flask import Flask, request, jsonify
import requests, os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server is running!", "ready": True})

VERIFY_TOKEN    = os.environ["VERIFY_TOKEN"]
WA_TOKEN        = os.environ["WA_TOKEN"]
PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"]
GROQ_API_KEY    = os.environ["GROQ_API_KEY"]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore():
    loader = TextLoader("docs.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

vectorstore = build_vectorstore()
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

def ask_rag(question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are a helpful assistant. Answer using only the provided context.

IMPORTANT LANGUAGE RULES:
- If user writes in Roman Urdu (Urdu written in English letters) reply in Roman Urdu
- If user writes in Urdu (Arabic script) reply in Urdu
- If user writes in English reply in English
- Always match the language of the user's message
- Keep answers simple, friendly and easy to understand

Context:
{context}

Question: {question}
Answer:"""
    return llm.invoke(prompt).content

@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        phone = msg["from"]
        text  = msg["text"]["body"]
        reply = ask_rag(text)
        send_whatsapp(phone, reply)
    except Exception as e:
        print("Error:", e)
    return jsonify({"status": "ok"})

def send_whatsapp(to, message):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to,
               "type": "text", "text": {"body": message}}
    requests.post(url, json=payload, headers=headers)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)