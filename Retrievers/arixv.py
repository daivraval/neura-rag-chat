'''A retriever takes a user query and returns the most relevant
documents or chunks from a database.

Now there are 2 types of retriever
1. by data source (Wikipedia, Arxiv, PubMed, etc.)
2. By retrieval strategy (Similarity, MMR, MultiQuery, etc.)'''


# By data source : 
from langchain_community.retrievers import ArxivRetriever

# create a retriever
retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_metadata=True
)

# query the retriever
docs= retriever.invoke("Large Language Models")

# print results
for i, docs in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {docs.metadata['title']}")
    print(f"Authors: {docs.metadata['authors']}")
    print(f"Abstract: {docs.metadata['summary']}")
    print(f"Summary",docs.page_content[:500])
    print("\n")

"""
Search Strategies

In most RAG (Retrieval-Augmented Generation) systems, retrievers mainly use three
core retrieval strategies. These are the ones you'll see most often in real projects and
tutorials.

1. Similarity Search (Most Common)
2. MMR (Max Marginal Relevance)
3. MultiQuery Retriever


1. Search Strategies - Similarity Search
The system compares the query vector with document vectors using similarity metrics
like:
· Cosine similarity (most common)
· Dot product
· Euclidean distance
The retriever finds the most similar vectors.
Top-K documents are retrieved
"""