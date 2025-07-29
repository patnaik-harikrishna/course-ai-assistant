import os
import pandas as pd
from llama_index.core.schema import Document
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
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

# Reading csv file containing various courses offered
def load_course_documents_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        # Format as readable text chunk
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

# Load documents
documents = load_course_documents_from_csv("./data/courses.csv")
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

        # Always use software courses prompt
        prompt = software_courses_prompt
        prompt_type = "software_courses"

        print(f"Selected prompt type: {prompt_type}")

        custom_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=OpenAI(temperature=0),
            text_qa_template=prompt
        )

        response = custom_query_engine.query(question)

        return {
            "answer": str(response),
            "contexts": contexts,
            "prompt_type": prompt_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
