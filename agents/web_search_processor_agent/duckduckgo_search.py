from ddgs import DDGS


class DuckDuckGoSearchAgent:
    """
    Performs web searches using DuckDuckGo API.
    Replaces Tavily search functionality with ddgs package.
    """

    def __init__(self):
        """
        Initialize the DuckDuckGo search agent.
        """
        pass

    def search_duckduckgo(self, query: str) -> str:
        """
        Perform a web search using DuckDuckGo API.

        Args:
            query: User search query

        Returns:
            Formatted search results as newline-separated string
            Format: title: X - url: Y - content: Z - score: S
        """
        try:
            # Strip any surrounding quotes from the query
            query = query.strip('"\'')

            # Initialize DDGS client
            ddgs = DDGS()

            # Perform text search (max 5 results)
            search_results = ddgs.text(query, max_results=5)

            # Check if results were found
            if not search_results or len(search_results) == 0:
                return "No relevant results found."

            # Format results to match Tavily output format
            formatted_results = []
            for result in search_results:
                # Extract fields from DDGS response
                # DDGS returns: title, href, body (and other metadata)
                title = result.get("title", "")
                url = result.get("href", "")
                content = result.get("body", "")

                # DDGS doesn't provide relevance scores, use default 0.5
                score = 0.5

                # Format as: title: X - url: Y - content: Z - score: S
                formatted_result = (
                    f"title: {title} - url: {url} - content: {content} - score: {score}"
                )
                formatted_results.append(formatted_result)

            # Return newline-separated results
            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error retrieving web search results: {e}"
