from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
#1. TEXT BASED SPLITTING ( SPLIT BY CHARACTER)
# splitter=CharacterTextSplitter(
#     separator="",
#     chunk_size=10,
#     chunk_overlap=1
# )
# data = TextLoader("1_document_loaders/notes.txt")
# docs=data.load()
# chunks = splitter.split_documents(docs)
# print(len(chunks))


# this outputs as 1.
# in langchain docs , it is mentioned that for a character based splitting "\n\n" needs to be used
# so , in the notes , leaving a blank line after a chunk makes it possible 

# template=ChatPromptTemplate.from_messages(
#     [("system", " You are an AI that summarises the text and doesnt leave any point"),
#      ("human","{data}")]
# )


'''this will load the pdf document in the form of a list 
for every page in the pdf file , one document is made in the list
[] is a list , and inside that () is a doc 
so the list looks like [(), (), (), .....]
print(docs[14])  prints the 15th page of the pdf file 

but this is for the small pdf file . What if we have 500 pages file ? '''

# splitter=CharacterTextSplitter(
#     separator = "",
#     chunk_size=1000,

# )


#TOKEN BASED SPLITTING : DIVIDES THE DOC BASED ON TOKENS ,SINCE LLMS PROCESS TOKENS INSTEAD OF WORDS
# THIS METHOD ALIGNS BETTER WITH HOW MODEL ACTUALLY READS THE TEXT  

