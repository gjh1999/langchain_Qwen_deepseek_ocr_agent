from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.runner import run_agent, run_agent_with_pdf
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return FileResponse("simple_pdf_chat_frontend.html")

@app.post("/chat")
async def chat(
    message: str = Form(""),
    file: UploadFile = File(None)
):
    try:
        if file:
            os.makedirs("temp", exist_ok=True)
            file_path = os.path.join("temp", file.filename)

            with open(file_path, "wb") as f:
                f.write(await file.read())

            answer = run_agent_with_pdf(file_path=file_path, question=message)
        else:
            answer = run_agent(message)

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"错误: {str(e)}"}