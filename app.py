import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS vector store safely
vector_store = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True  # Only enable if you trust the source
)

# Initialize the retriever
retriever = vector_store.as_retriever()

# Initialize the GPT model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit Interface
st.title("RAG-based Question-Answering System")
st.write("Upload documents (PDF, Word, CSV, or TXT), and ask questions based on their content.")

# Upload documents
uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "csv", "txt"], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to the documents folder
    for uploaded_file in uploaded_files:
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.success("Files uploaded successfully! Run the ingestion script to update the database.")

# User input for question
question = st.text_input("Ask a question:")

if question:
    # Generate answer using the RAG pipeline
    answer = qa_chain.run(question)
    st.write("Answer:", answer)
