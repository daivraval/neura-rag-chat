''' pdf -> chunks -> embeddings -> store in vectorDB 

 # user asks qn ^ related to that PDF
 #  -> embedding -> choose a chunk from vectorDB'''



from dotenv import load_dotenv

from neura import pipeline

load_dotenv()

retriever = pipeline.make_retriever(pipeline.open_index(), **pipeline.RETRIEVAL)
llm = pipeline.make_llm(max_new_tokens=300)
prompt = pipeline.make_prompt()

print("RAG system created")
print("Press 'quit' to exit")
while True:
    query=input("You:")
    if query=="quit":
        break
    docs=retriever.invoke(query)
    response = pipeline.generate(llm, prompt, query, docs)

    print(f"\nAI: {response}\n")

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