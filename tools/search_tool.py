from duckduckgo_search import DDGS


class WebSearchTool:
    
    def __init__(self, debug=False):
        self.debug = debug

    def search_duckduckgo(self, query, max_results=3):

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
    
    def search(self, query, max_results=3):
        """Interface method that matches the expected API in agent.py"""
        if self.debug:
            print(f"Searching for: {query}")
        return self.search_duckduckgo(query, max_results)