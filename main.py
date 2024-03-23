from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/chat', methods=['POST'])
def chat_with_pdf():
    data = request.get_json()
    user_question = data['question']

    if "conversation" not in app.config:
        app.config["conversation"] = None
    if "chat_history" not in app.config:
        app.config["chat_history"] = None

    if app.config["conversation"] is None:
        
        vectorstore = app.config["vectorstore"]
        app.config["conversation"] = get_conversation_chain(vectorstore)

    response = app.config["conversation"]({'question': user_question})
    app.config["chat_history"] = response['chat_history']

    messages = []
    for i, message in enumerate(app.config["chat_history"]):
        if i % 2 == 0:
            messages.append({'speaker': 'user', 'content': message.content})
        else:
            messages.append({'speaker': 'bot', 'content': message.content})

    return jsonify({'messages': messages})

if __name__ == '__main__':
    load_dotenv()
    pdf_docs = ["Sujet_corrige_EF_Analyse2_17-18.pdf"]
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    app.config["vectorstore"] = vectorstore
    app.run(debug=True)
