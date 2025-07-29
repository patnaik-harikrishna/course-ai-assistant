import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = openai_key

# Load documents
documents = SimpleDirectoryReader("./data").load_data()
if not documents:
    raise ValueError("No documents found in ./data directory")

print(f"Loaded {len(documents)} documents")

# Text splitting
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = text_splitter.get_nodes_from_documents(documents)
print(f"Split into {len(nodes)} chunks")

# Create index
index = VectorStoreIndex(nodes)

# Persist index
index.storage_context.persist(persist_dir="./storage")

# Retriever and query engine
retriever = index.as_retriever(similarity_top_k=3)
query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=OpenAI(temperature=0))

# FastAPI app
app = FastAPI()

# CORS middleware (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    question: str

# API endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        question = request.question
        nodes = retriever.retrieve(question)
        contexts = [node.get_content() for node in nodes]
        response = query_engine.query(question)
        return {
            "answer": str(response),
            "contexts": contexts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
