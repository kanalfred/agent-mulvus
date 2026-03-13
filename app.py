"""FastAPI web server for the RAG chat UI."""

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessagesTypeAdapter

from rag import agent

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    history: list = []


@app.get("/")
def index():
    return FileResponse("chat.html", headers={"Cache-Control": "no-store"})


@app.post("/chat")
def chat(req: ChatRequest):
    history = ModelMessagesTypeAdapter.validate_python(req.history) if req.history else None
    try:
        result = agent.run_sync(req.question, message_history=history)
    except ModelHTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "answer": result.output,
        "history": json.loads(result.all_messages_json()),
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = ModelMessagesTypeAdapter.validate_python(req.history) if req.history else None

    async def generate():
        try:
            async with agent.run_stream(req.question, message_history=history) as result:
                async for delta in result.stream_text(delta=True, debounce_by=None):
                    yield f"data: {json.dumps({'text': delta})}\n\n"
                history_data = json.loads(result.all_messages_json())
                yield f"event: done\ndata: {json.dumps({'history': history_data})}\n\n"
        except ModelHTTPError as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e), 'status': 502})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e), 'status': 500})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
