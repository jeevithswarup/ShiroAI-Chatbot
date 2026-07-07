from app.schemas import ChatRequest
from fastapi import FastAPI
from .services import stream_llm 
from app.schemas import ChatRequest
from fastapi.responses import StreamingResponse
from .services import stream_llm
app=FastAPI()

@app.post("/chat")
def chat(request: ChatRequest):

    return StreamingResponse(
        stream_llm(request.message),
        media_type="text/plain"
    )