import yfinance as yf
import pandas as pd


class StockDataFetcher:
    """
    A class to fetch historical stock data using Yahoo Finance.

    Methods:
        fetch_stock_data(symbol, start_date, end_date):
            Fetches historical stock data for a given symbol and date range.
    """

    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetches historical stock data for the specified symbol and date range.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL" for Apple, "TSLA" for Tesla).
            start_date (str): The start date in "YYYY-MM-DD" format.
            end_date (str): The end date in "YYYY-MM-DD" format.

        Returns:
            pd.DataFrame: A DataFrame containing the historical stock data.
        """
        try:
            # Fetch data using yfinance
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)

            # Check if data is returned
            if data.empty:
                print(f"No data found for {symbol} between {start_date} and {end_date}.")
                return pd.DataFrame()

            # Return the historical stock data
            return data

        except Exception as e:
            print(f"An error occurred while fetching stock data: {e}")
            return pd.DataFrame()


# Example Usage (TEST)
if __name__ == "__main__":
    # Instantiate the StockDataFetcher
    fetcher = StockDataFetcher()

    # Fetch historical stock data for Tesla for January 2024
    symbol = "TSLA"
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    stock_data = fetcher.fetch_stock_data(symbol, start_date, end_date)

    # Display the data
    if not stock_data.empty:
        print(stock_data)

