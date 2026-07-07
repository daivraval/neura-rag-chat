''' pdf -> chunks -> embeddings -> store in vectorDB 

 # user asks qn ^ related to that PDF
 #  -> embedding -> choose a chunk from vectorDB'''



import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
#from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vector_store=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
        }
)

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=300,
        do_sample=False,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )
)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context, say:
            "I could not find the answer in the document.""",
        ),
        (
            "human",
            """Context: {context}
            Question: {question}""",
        ),
    ]
)

print("RAG system created")
print("Press 'quit' to exit")
while True:
    query=input("You:")
    if query=="quit":
        break
    docs=retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")

# data = PyPDFLoader("1_document_loaders/PDF.pdf")
# docs=data.load()
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_documents(docs)
# template=ChatPromptTemplate.from_messages(
#     [("system"," you are an AI that summarizes the text"),
#      ("human","{data}")]
# )

# model = ChatMistralAI(model = "mistral-small-2506")
# result = model.invoke("Hello")
# print(result.content)

# We already have DB's like SQL, MongoDB , PostgreSQL, etc 
# Why do we need Vector DB's ?

'''
The biggest problem is this query 512 dimension embedding
is different from all the 1 lakh embeddings in our
database so we conduct a similarity search with all the
1 lakh embeddings. So you are working at 0(n) time complexity and you
are searching for 1 lakh time and then finding out and
this dataset can become more big so they are not the
reliable option for similarity searching.

Where as vector store use Approximate Nearest Neighbour
algorithms like -
a) HNSẀ

b) IVF (Inverted File Index) - Lets say we divide our Database in 5 clusters
using k-means or any other algo. so each will have 20000 embeddings and
all of them have some sort of similarity. Now we take average of each of 20000
clusters, called the centroid. So now , we have 5 average embeddings.
We compare the query embedding with the 5 centroid , the one with maximum similarity,
is taken and all the 20K embeddings from that cluster is taken to compare with the 
query embedding. So this , technically made it 5x times faster in terms of comparing
and finding the suitable embedding wrt the query embedding. 
USE CASE : Recommendation systems , AI search , RAG apps. 

c) PQ

Now there are many types of Vector stores : 
Chroma 
FAISS
Annoy

'''