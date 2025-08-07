import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.schema import Document
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from prompts import software_courses_prompt

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = openai_key

# ------------------------
# Lifespan startup handler
# ------------------------

retriever = None  # Declare globally

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever

    documents = load_course_documents_from_csv("./data/courses.csv")
    print(f"Loaded {len(documents)} documents")

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f"Split into {len(nodes)} chunks")

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir="./storage")

    retriever = index.as_retriever(similarity_top_k=3)
    print("Retriever is ready")

    yield  # Runs the app

# ------------------------
# Create app with lifespan
# ------------------------

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Data loader function
# ------------------------

def load_course_documents_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Title: {row['title']}\n"
            f"Category: {row['category']}\n"
            f"Level: {row['level']}\n"
            f"Duration: {row['duration']}\n"
            f"Certification: {row['certification']}\n"
            f"Price: {row['price']}\n"
            f"Description: {row['description']}\n"
        )
        documents.append(Document(text=text))

    return documents

# ------------------------
# API schema and endpoint
# ------------------------

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        question = request.question
        print("Received question:", question)

        nodes = retriever.retrieve(question)
        print(f"Retrieved {len(nodes)} nodes")

        contexts = [node.get_content() for node in nodes]
        prompt = software_courses_prompt

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=OpenAI(temperature=0),
            text_qa_template=prompt
        )

        response = query_engine.query(question)

        return {
            "answer": str(response),
            "contexts": contexts,
            "prompt_type": "software_courses"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))