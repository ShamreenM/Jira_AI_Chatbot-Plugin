from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from JiraAIChatbot4 import retrieve_docs

app = FastAPI()

# Allow Forge (Atlassian) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    api_key: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = retrieve_docs(request.query, request.api_key)
    return {"answer": result}

@app.get("/health")
async def health():
    return {"status": "ok"}