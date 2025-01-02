import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from DataProcessor import DataProcessor
import pandas as pd
import os
import joblib  # For saving and loading the scaler


class ModelTrainer:
    """
    A class to train a model that predicts stock movement based on sentiment and stock data.
    """

    def __init__(self, input_dim, model_save_path="stock_movement_model"):
        """
        Initializes the ModelTrainer.

        Args:
            input_dim (int): Number of input features.
            model_save_path (str): Path to save the trained model.
        """
        self.model_save_path = model_save_path
        self.model = self._build_model(input_dim)
        self.scaler_save_path = os.path.join(model_save_path, "scaler.pkl")

    def _build_model(self, input_dim):
        """
        Builds a feed-forward neural network for stock movement prediction.

        Args:
            input_dim (int): Number of input features.

        Returns:
            tf.keras.Model: Compiled model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),  # Regularization
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")  # For binary classification (price up or down)
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def preprocess_data(self, combined_data, target_column="target"):
        """
        Preprocesses the data by normalizing numerical features and splitting into train/test sets.

        Args:
            combined_data (pd.DataFrame): Combined DataFrame with sentiment and stock features.
            target_column (str): Column name for the target variable.

        Returns:
            tuple: Train/test splits for features (X) and target (y).
        """
        features = ["sentiment_score", "open", "high", "low", "close", "volume", "moving_avg"]

        # Normalize features
        scaler = MinMaxScaler()
        combined_data[features] = scaler.fit_transform(combined_data[features])

        # Save the scaler for future predictions
        joblib.dump(scaler, self.scaler_save_path)

        # Split features and target
        X = combined_data[features].values
        y = combined_data[target_column].values

        # Train-test split
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, combined_data, target_column="target", epochs=10, batch_size=16):
        """
        Trains the model using the provided data.

        Args:
            combined_data (pd.DataFrame): Combined DataFrame with sentiment and stock features.
            target_column (str): Column name for the target variable.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(combined_data, target_column)

        # Train the model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )

        # Evaluate the model
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))

    def save_model(self):
        """
        Saves the trained model in .keras format.
        """
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Define the full path with the .keras extension
        save_path = os.path.join(self.model_save_path, "model.keras")

        # Save the model
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    def predict(self, new_data):
        """
        Makes predictions on new data.

        Args:
            new_data (pd.DataFrame): DataFrame with new features.

        Returns:
            np.array: Predicted probabilities for upward stock movement.
        """
        # Load the scaler
        scaler = joblib.load(self.scaler_save_path)

        # Normalize the new data
        features = ["sentiment_score", "open", "high", "low", "close", "volume", "moving_avg"]
        new_data[features] = scaler.transform(new_data[features])

        # Make predictions
        return self.model.predict(new_data[features].values)


# Example Usage (TEST)
if __name__ == "__main__":
    # Sample sentiment data
    sentiment_data = pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "sentiment_score": [0.85, -0.75, 0.0]
    })

    # Sample stock data
    stock_data = pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [150, 145, 140],
        "high": [155, 150, 145],
        "low": [148, 143, 138],
        "close": [153, 148, 141],
        "volume": [100000, 120000, 110000]
    })

    # Combine sentiment and stock data using DataProcessor
    data_processor = DataProcessor()
    combined_data = data_processor.combine_with_stock_data(stock_data, sentiment_data)

    # Add a target column (e.g., 1 if close > open, otherwise 0)
    combined_data["target"] = (combined_data["close"] > combined_data["open"]).astype(int)

    # Initialize and train ModelTrainer
    trainer = ModelTrainer(input_dim=7)  # 7 features: sentiment_score, open, high, low, close, volume, moving_avg
    trainer.train(combined_data)

    # Save the trained model
    trainer.save_model()

    # Predict using the saved model
    new_data = combined_data.copy()  # Simulate new data
    predictions = trainer.predict(new_data)
    print("Predictions:", predictions)
