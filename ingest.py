import openai
import os
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")

# Load PDF and split into chunks
loader = PyPDFLoader("data.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitted_docs = text_splitter.split_documents(documents)

texts = [doc.page_content for doc in splitted_docs]

# Generate embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))

# Load Qdrant credentials from .env
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize vector store
qdrant = QdrantVectorStore.from_texts(
    texts=texts,
    embedding=embeddings_model,
    url=qdrant_url,
    api_key=qdrant_api_key,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")
