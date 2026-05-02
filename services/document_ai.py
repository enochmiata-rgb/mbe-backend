from pathlib import Path
import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract full text from PDF.
    """
    text_parts = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)

    return "\n".join(text_parts)


def analyze_document_text(text: str) -> dict:
    """
    Simple strategic extraction (version 1).
    """

    text_lower = text.lower()

    result = {
        "production_mentions": 0,
        "revenue_mentions": 0,
        "risk_mentions": 0,
        "keywords": [],
    }

    if "production" in text_lower:
        result["production_mentions"] += text_lower.count("production")

    if "revenue" in text_lower or "revenu" in text_lower:
        result["revenue_mentions"] += text_lower.count("revenue")
        result["revenue_mentions"] += text_lower.count("revenu")

    if "risk" in text_lower or "risque" in text_lower:
        result["risk_mentions"] += text_lower.count("risk")
        result["risk_mentions"] += text_lower.count("risque")

    # simple keyword extraction
    important_words = [
        "production",
        "oil",
        "gas",
        "revenue",
        "market",
        "risk",
        "investment",
        "strategy",
    ]

    result["keywords"] = [
        w for w in important_words if w in text_lower
    ]

    return result


def analyze_pdf(file_path: str) -> dict:
    text = extract_text_from_pdf(file_path)
    return analyze_document_text(text)