from duckduckgo_search import DDGS


def search_duckduckgo(query, max_results=3):
    """Perform a DuckDuckGo search for the given query."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r.get('title', 'No title')}\nSource: {r.get('href', 'No source')}\n{r['body']}")
        if results:
            return "\n\n".join(results)
        else:
            return "No relevant information found."
    except Exception as e:
        return f"Search error: {str(e)}"