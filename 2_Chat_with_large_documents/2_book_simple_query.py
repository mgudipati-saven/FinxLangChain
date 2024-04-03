from pinecone import Pinecone
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings_api = OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY"))
pinecone_index = "langchain-vector-store"
pc = Pinecone(api_key=config("PINECONE_API_KEY"))
index = pc.Index(pinecone_index)
vectorstore = PineconeVectorStore(index, embeddings_api, "text")
query = "What is the fastest way to get rich?"
print(vectorstore.similarity_search(query, k=5))

