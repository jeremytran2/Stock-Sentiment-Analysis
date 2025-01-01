import requests
import json

class SocialMediaFetcher:
    """
    A class to fetch social media posts using the Twitter API.

    Attributes:
        api_key (str): The Bearer Token for accessing the Twitter API.
        base_url (str): The base URL for the Twitter API.
    """

    def __init__(self, config_path="config.json"):
        """
        Initializes the SocialMediaFetcher with the given API key.

        Args:
            config_path (str): Your Twitter API Bearer Token.
        """
        # Load API key from configuration file
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        self.api_key = config["x-developer-portal-api-key"]
        self.base_url = "https://api.twitter.com/2/tweets/search/recent"

    def fetch_posts(self, query, max_results=10):
        """
        Fetches social media posts based on the given query.

        Args:
            query (str): The search keyword or hashtag.
            max_results (int): Maximum number of posts to fetch (up to 100).

        Returns:
            list: A list of social media posts or an empty list if none are found.
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        params = {
            'query': query,
            'max_results': max_results,
            'tweet.fields': 'created_at,lang,text'
        }

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            return data.get('data', [])
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []



