from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil
import uuid

from services.document_ai import analyze_pdf

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/api/documents/analyze")
async def analyze_uploaded_document(file: UploadFile = File(...)):
    """
    Upload a PDF and run AI analysis.
    """

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analysis = analyze_pdf(str(file_path))

    return {
        "file": file.filename,
        "analysis": analysis,
    }