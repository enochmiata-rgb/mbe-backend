import requests
from typing import Tuple


def download_pdf(pdf_url: str) -> Tuple[bytes, str]:
    """
    Télécharge un PDF et retourne :
    - contenu binaire
    - content-type
    """

    response = requests.get(pdf_url, allow_redirects=True, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "application/pdf" not in content_type.lower():
        raise ValueError("URL does not return a PDF")

    return response.content, content_type