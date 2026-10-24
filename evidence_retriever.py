
import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "LLMResponseQualityChecker/1.0 (educational project)"
}


def retrieve_evidence(query, max_results=3):
    """Search Wikipedia and return article titles, URLs, and extracts."""

    query = query.strip()

    if not query:
        return []

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
    }

    response = requests.get(
        WIKIPEDIA_API,
        params=search_params,
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()

    search_results = response.json().get(
        "query", {}
    ).get("search", [])

    evidence_items = []

    for item in search_results:
        title = item.get("title", "")

        page_params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exchars": 3500,
            "titles": title,
            "format": "json",
        }

        page_response = requests.get(
            WIKIPEDIA_API,
            params=page_params,
            headers=HEADERS,
            timeout=20,
        )
        page_response.raise_for_status()

        pages = page_response.json().get(
            "query", {}
        ).get("pages", {})

        page = next(iter(pages.values()), {})
        extract = page.get("extract", "").strip()
        page_id = page.get("pageid")

        if extract and page_id:
            evidence_items.append({
                "title": title,
                "url": (
                    "https://en.wikipedia.org/"
                    f"?curid={page_id}"
                ),
                "extract": extract,
            })

    return evidence_items

