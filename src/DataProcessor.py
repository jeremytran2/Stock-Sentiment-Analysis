import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    """
    A class to preprocess data for sentiment analysis and stock prediction.
    """

    def __init__(self, max_words=10000, max_len=100):
        """
        Initializes the DataProcessor with tokenizer and configuration.

        Args:
            max_words (int): Maximum number of words for the tokenizer.
            max_len (int): Maximum sequence length for padding.
        """
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.max_len = max_len
        self.embedder = pipeline("feature-extraction", model="bert-base-uncased")

    def clean_text(self, text):
        """
        Cleans raw text by removing URLs, mentions, hashtags, and special characters.

        Args:
            text (str): Raw text to clean.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
        return text.lower().strip()

    def tokenize_and_pad(self, texts):
        """
        Tokenizes and pads sequences for model input.

        Args:
            texts (list): List of cleaned text strings.

        Returns:
            ndarray: Tokenized and padded sequences.
        """
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding="post")

    def embed_text(self, texts):
        """
        Converts text into embeddings using a pre-trained model (BERT).

        Args:
            texts (list): List of cleaned text strings.

        Returns:
            list: List of embeddings for each text.
        """
        return [self.embedder(text)[0] for text in texts]

    def combine_with_stock_data(self, stock_data, sentiment_data):
        """
        Combines sentiment data with stock price data.

        Args:
            stock_data (pd.DataFrame): DataFrame with stock prices.
            sentiment_data (pd.DataFrame): DataFrame with sentiment scores.

        Returns:
            pd.DataFrame: Combined dataset.
        """
        # Merge on timestamp
        combined = pd.merge(stock_data, sentiment_data, on="timestamp", how="inner")

        # Add technical indicators
        combined["moving_avg"] = combined["close"].rolling(window=10).mean()
        combined["sentiment_score"] = combined["sentiment"].map({"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0})

        # Normalize data
        scaler = MinMaxScaler()
        combined[["open", "high", "low", "close", "volume"]] = scaler.fit_transform(
            combined[["open", "high", "low", "close", "volume"]]
        )
        return combined

    def save_preprocessed_data(self, data, file_path="preprocessed_data.csv"):
        """
        Saves preprocessed data to a CSV file.

        Args:
            data (pd.DataFrame): Preprocessed dataset.
            file_path (str): File path to save the dataset.
        """
        data.to_csv(file_path, index=False)



