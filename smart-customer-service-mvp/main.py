from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agents import process_message
import uvicorn

app = FastAPI(title="智能客服多Agent系统")

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    reply = await process_message(req.message, req.session_id)
    return JSONResponse(content={"reply": reply})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)