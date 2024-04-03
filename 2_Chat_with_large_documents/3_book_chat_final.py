from pinecone import Pinecone
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

openai_key = config("OPENAI_API_KEY")
davinci_api = OpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-3.5-turbo-instruct")
embeddings_api = OpenAIEmbeddings(openai_api_key=openai_key)

pinecone_index = "langchain-vector-store"
pc = Pinecone(api_key=config("PINECONE_API_KEY"))
index = pc.Index(pinecone_index)
vectorstore = PineconeVectorStore(index, embeddings_api, "text")
qa_chain = load_qa_chain(davinci_api, chain_type="stuff")

def ask_question_to_book(question: str, verbose=False) -> str:
    matching_pages = vectorstore.similarity_search(question, k=5)
    if verbose:
        print(f"Matching Documents:\n{matching_pages}\n")
    result = qa_chain.run(input_documents=matching_pages, question=question)
    print(result, "\n")
    return result

ask_question_to_book("What is the fastest way to get rich?", verbose=False)
ask_question_to_book("What is the problem with most people?")
ask_question_to_book("What is the best way to peel bananas?")
