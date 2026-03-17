import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.services.fitness_assistant import FitnessAssistant

assistant_instance = FitnessAssistant()


def chat(message: str, history: list) -> str:
    """Trimite mesajul utilizatorului catre asistentul fitness si returneaza raspunsul."""
    try:
        return assistant_instance.assistant_response(message)
    except Exception as e:
        return f"Eroare: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="Asistent Fitness 🏋️‍♂️",
    description=(
        "Bun venit! Sunt antrenorul tau virtual de fitness. "
        "Intreaba-ma despre exercitii, planuri de antrenament, grupe musculare "
        "sau cum sa iti structurezi antrenamentele."
    ),
    textbox=gr.Textbox(placeholder="Scrie mesajul tau aici...", container=False, scale=7),
    examples=[
        "Ce exercitii pot face pentru biceps acasa, fara echipament?",
        "Cum sa slabesc si sa ard grasime eficient?",
        "Propune-mi un plan de antrenament pentru incepatori.",
        "Care sunt cele mai bune exercitii pentru spate?",
    ],
    theme=gr.themes.Soft(),
    type="messages",
)

app = FastAPI()


@app.get("/health.html", response_class=HTMLResponse)
def health() -> HTMLResponse:
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Health</title></head>
<body><h1>OK</h1><p>Fitness Assistant is running.</p></body>
</html>""")


app = gr.mount_gradio_app(app, demo, path="")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
