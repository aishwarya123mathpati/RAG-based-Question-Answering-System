import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_documents(directory_path):
    """Loads and processes documents from various formats."""
    loaders = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if file_name.endswith(".txt"):
            loaders.append(TextLoader(file_path))
        elif file_name.endswith(".pdf"):
            loaders.append(PyPDFLoader(file_path))
        elif file_name.endswith(".csv"):
            loaders.append(CSVLoader(file_path))
        elif file_name.endswith(".docx"):
            loaders.append(UnstructuredWordDocumentLoader(file_path))

    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """Creates and saves a FAISS vector store from document embeddings."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")  # Save the new index to 'faiss_index' folder
    print("Vector store created and saved!")

    return vector_store

if __name__ == "__main__":
    # Load and chunk documents from the specified directory
    documents = load_documents("documents/")
    chunked_documents = chunk_documents(documents)

    # Create and save the vector store
    create_vector_store(chunked_documents)
