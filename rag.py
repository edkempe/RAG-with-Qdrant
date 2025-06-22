import openai
import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print("##############")

collection_name = "vector_db"
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))
db = QdrantVectorStore(client=client, collection_name="vector_db", embedding=embeddings_model)

llm = OpenAI(openai_api_key=os.getenv("OPENAI_KEY"))

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

def ask_question(question):
    result = retrieval_qa.invoke({"query": question})
    return result['result']

if __name__ == "__main__":
    question = "What is the purpose of this document?"
    answer = ask_question(question)
    print("Question:", question)
    print("Answer:", answer)
