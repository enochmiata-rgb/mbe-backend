import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict


def find_pdf_links(page_url: str) -> List[Dict]:
    """
    Trouve tous les liens PDF d'une page HTML
    Supporte :
    - .pdf
    - /attachment/
    - URLs relatives
    - redirections
    """

    results = []

    try:
        response = requests.get(page_url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return [{"error": str(e)}]

    soup = BeautifulSoup(response.text, "html.parser")

    links = soup.find_all("a", href=True)

    for link in links:
        href = link["href"]
        full_url = urljoin(page_url, href)

        # détecte pdf ou attachment
        if ".pdf" in href.lower() or "/attachment/" in href.lower():

            try:
                head = requests.head(full_url, allow_redirects=True, timeout=10)
                content_type = head.headers.get("Content-Type", "")

                if "application/pdf" in content_type.lower():
                    results.append({
                        "url": full_url,
                        "contentType": content_type,
                        "status": "valid_pdf"
                    })
                else:
                    results.append({
                        "url": full_url,
                        "contentType": content_type,
                        "status": "not_pdf"
                    })

            except Exception as e:
                results.append({
                    "url": full_url,
                    "error": str(e),
                    "status": "error"
                })

    return results