import shutil
import os
from fastapi import FastAPI, UploadFile, File
from app.rag import index_pdf
from app.vector_db import semantic_search
from app.schemas import ChatRequest
from .services import stream_llm ,chat
from app.schemas import ChatRequest
from fastapi.responses import StreamingResponse #type: ignore



app=FastAPI()

from app.services import chat

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        chat(
            request.conversation_id,
            request.message
        ),
        media_type="text/plain"
    )



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("app/documents", exist_ok=True)

    file_path = f"app/documents/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    index_pdf(file_path)

    return {"message": "PDF indexed successfully"}