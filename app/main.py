from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# reduce TensorFlow log noise in server output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from src.services.fitness_assistant import FitnessAssistant

from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):    
    load_dotenv()
    yield

app = FastAPI(lifespan=lifespan)

# instanta chatbotului FitnessAssistant
assistant_instance = FitnessAssistant()

@app.get("/")
async def root():    
    return {"message": "Salut, Fitness Assistant ruleaza!"}

@app.get("/health.html", response_class=HTMLResponse)
async def health_check():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Check</title>
    </head>
    <body>
        <h1>OK</h1>
        <p>Service is healthy</p>
        <p>Status: Running</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    """
    Endpoint principal de chat:
    Trimite mesajul catre FitnessAssistant.assistant_response()
    si returneaza raspunsul.
    """
    try:
        # ruleaza metoda blocanta intr-un thread separat
        response = await asyncio.wait_for(
            asyncio.to_thread(assistant_instance.assistant_response, request.message),
            timeout=45
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Raspunsul de chat a expirat")
    except Exception as e:
        logging.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat esuat: {repr(e)}")

    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
