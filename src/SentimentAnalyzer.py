from transformers import pipeline
import pandas as pd

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using a pre-trained model.
    """

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the SentimentAnalyzer with a pre-trained sentiment model.

        Args:
            model_name (str): Name of the pre-trained sentiment analysis model.
        """
        self.analyzer = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, texts):
        """
        Performs sentiment analysis on a list of texts.

        Args:
            texts (list): List of text strings to analyze.

        Returns:
            list: List of dictionaries containing sentiment results.
        """
        return self.analyzer(texts)

    def add_sentiment_to_dataframe(self, df, text_column="text"):
        """
        Adds sentiment results to a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing a column of text data.
            text_column (str): Name of the column containing text data.

        Returns:
            pd.DataFrame: Updated DataFrame with sentiment results.
        """
        sentiments = self.analyze(df[text_column].tolist())
        df["sentiment"] = [result["label"] for result in sentiments] # Add the sentiment to data frame (POSITIVE, NEGATIVE OR NEUTRAL)
        df["sentiment_score"] = [result["score"] for result in sentiments] # Add the sentiment score to df
        return df

# Example Usage (TEST)
if __name__ == "__main__":
    # Example fetched data
    data = pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "text": [
            "Tesla stock is performing really well today!",
            "Market crashes are making investors nervous.",
            "Neutral perspective on Tesla's current situation."
        ]
    })

    # Initialize SentimentAnalyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Perform sentiment analysis
    analyzed_data = sentiment_analyzer.add_sentiment_to_dataframe(data, text_column="text")

    # Show all columns
    pd.set_option("display.max_columns", None)

    # Print the results
    print(analyzed_data)
