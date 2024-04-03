import re
from pinecone import Pinecone
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore

embeddings_api = OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY"))
pinecone_index = "langchain-vector-store"
pc = Pinecone(api_key=config("PINECONE_API_KEY"))
index = pc.Index(pinecone_index)

loader = PyPDFLoader("data/How-to-succeed.pdf")
data: list[Document] = loader.load_and_split()
page_texts: list[str] = [page.page_content for page in data]
page_texts_fixed: list[str] = [re.sub(r"\t|\n", " ", page) for page in page_texts]

vector_database = PineconeVectorStore(
    index_name=pinecone_index, embedding=embeddings_api, pinecone_api_key=config("PINECONE_API_KEY")
)
vector_database.add_texts(page_texts_fixed)