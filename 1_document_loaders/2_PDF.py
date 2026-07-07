#  2.TOKEN BASED TEXT SPLITTING
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data= PyPDFLoader("1_document_loaders/PDF.pdf")
docs= data.load()
splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunks=splitter.split_documents(docs)
print(len(chunks))


# 3.SPLITTING RECURSIVELY

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
data= PyPDFLoader("1_document_loaders/PDF.pdf")
docs= data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)   
chunks=splitter.split_documents(docs)
print(len(chunks))

