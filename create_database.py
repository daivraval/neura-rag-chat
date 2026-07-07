#load pdf
#split into chunks
#create the embeddings
#store into chroma
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("1_document_loaders/PDF.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
chunk_size = 1000,
chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

# Remove invalid Unicode surrogate characters that the PDF extraction leaves
# behind, otherwise the tokenizer rejects them.
for chunk in chunks:
    chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db" 
)