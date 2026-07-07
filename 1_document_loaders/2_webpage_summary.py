from langchain_community.document_loaders import WebBaseLoader
url = "https://www.apple.com/in/macbook-pro/"
data=WebBaseLoader(url)
docs=data.load()
print(docs[0].page_content)

'''
Chunking : Splitting a Pdf file into multiple chunks in order for the LLM to be contetx aware

 1. Most of the embedding models and language model have a context 
 window and they cannot take infinite tokens at the same time so
 for that we have to use the text splitting. 

2. Better Retrieval in RAG
 In Retrieval-Augmented Generation (RAG), we search through vector
 embeddings. If the document is too large:
 embeddings become less precise
 Smaller chunks allow the system to retrieve only the most relevant piece
 of information instead of the whole document.

 3. More Accurate Embeddings
 Embeddings work best when the text represents one clear idea or topic.
 If you embed very large text:
 . multiple topics mix together
 · semantic meaning becomes blurred
 Chunking ensures each embedding represents a focused concept.

 4. Faster Processing
 Working with smaller chunks:
 · speeds up embedding creation
 · speeds up similarity search in vector databases (Chroma, Pinecone,
 FAISS).
 Large documents would make the pipeline slower and expensive.

 We are going to see 3 types of text splitting methods.

 1. Character - Based Splitting
 2. Token-Based Splitting
 3. Semantic / Meaning-Based Splitting

 1. Character - Based Splitting

 Character-based splitting is the simplest method of text splitting.
 It divides a large document into smaller chunks based on the number of
 characters, without understanding words, tokens, or meaning.'''


