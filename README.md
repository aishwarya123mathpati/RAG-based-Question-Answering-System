# RAG-based-Question-Answering-System
RAG-based Question Answering System
This project implements a Retrieval-Augmented Generation (RAG) system to answer questions based on the content of documents (PDF, Word, CSV, and TXT files). It uses the FAISS library for efficient document vector search and OpenAI's GPT model for generating answers.

The project is divided into two main components:

Data Ingestion: Loads, processes, and chunks documents into smaller sections, then creates a FAISS vector store.
Streamlit Application: Provides a web interface where users can upload documents and ask questions based on their contents.

# Features
- Upload PDF, Word, CSV, and TXT documents.
- The documents are processed and converted into vector embeddings using OpenAI Embeddings.
- The system stores the embeddings in a FAISS vector store for efficient retrieval.
- Users can ask questions, and the system generates answers using GPT-3.5 Turbo model based on the uploaded documents.

# Requirements
- Python 3.7+
- Required packages:
- langchain_community
- faiss-cpu
- openai
- streamlit
- python-dotenv

# Setup

- Clone the repository
- cd RAG-based_Question_Answering_System
- Set up environment variables(your openAI API key)
- Install dependencies
- Run the data ingestion script
- Start the Streamlit Application
- Upload Documents
- Ask questions
