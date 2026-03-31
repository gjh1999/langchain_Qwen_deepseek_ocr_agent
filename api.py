from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import tempfile
import os

from ocr_pdf_service import init_model, ocr_pdf

app = FastAPI(title="DeepSeek OCR PDF API")


@app.on_event("startup")
def startup_event():
    init_model()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr_pdf")
async def ocr_pdf_api(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF is supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        output_dir = os.path.join(tmpdir, "output")

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = ocr_pdf(pdf_path, output_dir)

        return JSONResponse(
            {
                "filename": file.filename,
                "markdown": result["markdown"],
                "markdown_det": result["markdown_det"],
                "mmd_path": result["mmd_path"],
                "mmd_det_path": result["mmd_det_path"],
                "layout_pdf_path": result["layout_pdf_path"],
            }
        )