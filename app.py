import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from qdrant_client import QdrantClient
import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = splitter.split_text(text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def get_conversation_chain(vectorStore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userquestion(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    qdrant_client = QdrantClient(
        url="https://c1ea3a70-e824-4e28-a539-b425a95829ae.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key = os.getenv("QDRANT_API_KEY")
    )

    st.set_page_config(page_title="Chat with Multiple PDF's", page_icon=":books:", layout="wide")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDF's :books:")
    user_question = st.text_input("Ask a question to get started")

    if user_question:
        handle_userquestion(user_question)


    with st.sidebar:
        st.subheader("Your docs")
        pdf_docs = st.file_uploader("Upload PDF and click on process", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorStore = get_vector_store(text_chunks)
                
                st.session_state.conversation =  get_conversation_chain(vectorStore)







if __name__ == "__main__":
    main()