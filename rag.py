import openai
import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_KEY not set")

# Load Qdrant credentials from .env
qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    raise ValueError("QDRANT_URL not set")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY not set")

# Initialize Qdrant client with API key
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    prefer_grpc=False
)

collection_name = "vector_db"
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))

# Connect to existing Qdrant collection
db = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings_model
)

# Set up the LLM and RetrievalQA chain
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
