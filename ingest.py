import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PDF_DIR = "/content/data"
VECTORSTORE_DIR = "vectorstore"

def load_pdfs():
    documents = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
            documents.extend(loader.load())
    return documents

def main():
    # Load PDFs
    documents = load_pdfs()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # ✅ GOOGLE GEMINI EMBEDDINGS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ) 

    # Create or load vector store
    if os.path.exists(VECTORSTORE_DIR):
        vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings)
        
        # Add only new documents
        existing_ids = {doc.metadata.get("id") for doc in vectorstore.docstore._dict.values()}
        new_docs = [doc for doc in texts if doc.metadata.get("id") not in existing_ids]
        if new_docs:
            vectorstore.add_documents(new_docs)
            vectorstore.save_local(VECTORSTORE_DIR)
    else:
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTORSTORE_DIR)


if __name__ == "__main__":
    main()