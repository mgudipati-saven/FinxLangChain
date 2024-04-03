from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from decouple import config

prompt = ChatPromptTemplate.from_template(
    "Please tell me which foods are famous in {place}"
)
model = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"))
chain = prompt | model

result = chain.invoke({"place": "Hyderabad"})
print(result.content)