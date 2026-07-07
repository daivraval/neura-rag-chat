"""
2.Search Strategies - MMR (maximum marginal relevance)
Lets understand this with an example :
Imagine you are building a RAG system for your students.
Your knowledge base contains many chunks about Gradient Descent.
Your database chunks look like this:
Chunk 1: Gradient descent is an optimization algorithm used in machine learning.
Chunk 2: Gradient descent minimizes the loss function.
Chunk 3: Gradient descent is an optimization that minimizes the loss function.
Chunk 4: Neural networks use gradient descent for training.
Chunk 5: Support Vector Machines are supervised learning algorithms.
Now the user asks
What is Gradient Descent ?
Normal Similarity search finds the most similar chunks to the query embedding.
and gives chunk 1, chunk 2 , chunk 3
but if you see carefully all the 3 chunks are saying the same thing.
This wastes:
· context window
· token usage
· information diversity
The LLM doesn't learn new information.
MMR stands for:
Max Marginal Relevance
Its goal is to balance two things:
1. Relevance to the query
2. Diversity among retrieved documents
So instead of retrieving similar documents, it retrieves different but relevant documents.
"""
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma.from_documents(docs, embeddings)


similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n===== MMR Results =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)

'''
Similarity Search Results

Gradient descent is an optimization algorithm used in machine learning.
Gradient descent is an optimization that minimizes the loss function.
Gradient descent minimizes the loss function.

MMR Results

Gradient descent is an optimization algorithm used in machine learning.
Gradient descent minimizes the loss function.
Support Vector Machines are supervised learning algorithms.
'''