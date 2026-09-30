#load pdf
#split into chunks
#create the embeddings
#store into chroma
from dotenv import load_dotenv

from neura import pipeline

load_dotenv()

chunks = pipeline.load_chunks()
vectorstore = pipeline.build_index(chunks, persist_dir=pipeline.CHROMA_DIR)
print(f"Indexed {len(chunks)} chunks into {pipeline.CHROMA_DIR}")
