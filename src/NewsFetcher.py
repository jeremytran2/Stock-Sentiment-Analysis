import requests
import json
import pandas as pd

class NewsFetcher:
    """
    A class to fetch financial news articles using the NewsAPI.

    Attributes:
        api_key (str): The API key for accessing NewsAPI.
        base_url (str): The base URL for the NewsAPI.
    """

    def __init__(self, config_path="config.json"):
        """
        Initializes the NewsFetcher with the given API key.

        Args:
            config_path (str): path to config.json that contains News API key.
        """
        # Load API key from configuration file
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        self.api_key = config["news_api_key"]
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query, from_date=None, to_date=None, language="en", page_size=None):
        """
        Fetches news articles based on the given query and filters.

        Args:
            query (str): The search keyword or topic (e.g., "Tesla", "AAPL").
            from_date (str): The starting date for the news (YYYY-MM-DD). Default is None.
            to_date (str): The ending date for the news (YYYY-MM-DD). Default is None.
            language (str): The language of the articles (default is "en").
            page_size (int): Number of articles to fetch per page
        Returns:
            pd.DataFrame: A DataFrame of news articles with a standardized timestamp column.
        """
        # Define request parameters
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": language,
            "pageSize": page_size,
            "apiKey": self.api_key,
        }

        try:
            # Make the API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            data = response.json()

            # Check if the request was successful
            if data.get("status") == "ok":
                articles = data.get("articles", [])

                # Convert to DataFrame
                if articles:
                    df = pd.DataFrame(articles)

                    # Convert 'publishedAt' to 'timestamp' and format the date
                    df["timestamp"] = pd.to_datetime(df["publishedAt"]).dt.date
                    df.drop(columns=["publishedAt"], inplace=True)  # Drop the original 'publishedAt' column

                    return df
                else:
                    print("No articles found.")
                    return pd.DataFrame()  # Return an empty DataFrame if no articles are found
            else:
                print(f"Error: {data.get('message')}")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()

    def clean_news(self, articles):
        """
        Cleans the fetched news articles by extracting relevant fields.

        Args:
            articles (list): A list of raw news articles (dictionaries).

        Returns:
            list: A list of cleaned articles containing only relevant fields.
        """
        cleaned_articles = []
        for article in articles:
            cleaned_articles.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "source": article.get("source", {}).get("name"),
            })
        return cleaned_articles



#Example usuage (TEST)
if __name__ == "__main__":
    # Instantiate the NewsFetcher
    news_fetcher = NewsFetcher(config_path="config.json")

    # Fetch news about Tesla
    articles = news_fetcher.fetch_news(
        query="Tesla",
        language="en",
        page_size=30
    )

    # # Print the articles
    # for article in articles:
    #     print(f"Title: {article['title']}")
    #     print(f"Published At: {article['publishedAt']}")
    #     print(f"Source: {article['source']['name']}")
    #     print("-----")

    print(articles.to_string())
    print(news_fetcher.fetch_news(query="Tesla", from_date="2024-12-03", to_date="2024-12-31"))

