"""
3. Search Strategies - MultiQuery
Imagine you built a RAG system for machine learning notes.
Your database contains these chunks:
Chunk 1: Gradient descent is an optimization algorithm used to minimize loss functions.
Chunk 2: Neural networks are trained using optimization algorithms like gradient descent.
Chunk 3: Weight updates in machine learning models are done using gradient descent.
Chunk 4: Support Vector Machines are supervised learning algorithms.
Now the user asks:
"What is gradient descent?"
the retriever might return:
Chunk 1
Chunk 3
But notice something
There are other relevant chunks like Chunk 2 that might not be retrieved.
Why?
Because the query wording is limited.

The Core Problem
A single query might not capture all ways a concept can be expressed.
for example : "What is gradient descent?" 
could be asked as
"Explain gradient descent"
"How does gradient descent work?"
"Optimization algorithm used in neural networks"
Different wording > different embeddings > different search results.
This is where MultiQuery Retriever helps.

So Instead of using one query, it generates multiple variations of the query using
an LLM. so now the pipeline becomes
User Query -> LLM generates multiple queries-> Retriever searches for each query->Combine all retrieved documents
"""
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv



load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()


llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        task="text-generation",
        max_new_tokens=300,
        do_sample=False,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)