import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
import time
import pandas as pd
from typing import List, Tuple, Dict
import os
import sys
import pickle
import json


class ClassificationMetrics:
    def __init__(self, accuracy: float, precision: float, recall: float, f1_score: float):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score


class SpamClassifier:
    def __init__(self):
        self.vocabulary = [
        "win", "free", "money", "click", "offer",
        "buy", "urgent", "now", "limited", "congratulations",
        "prize", "winner", "selected", "claim", "gift",
        "card", "verify", "account", "secure", "payment",
        "transfer", "bank", "password", "login", "credit",
        "loan", "mortgage", "investment", "profit", "earn",
        "income", "wealth", "rich", "million", "billion",
        "lottery", "jackpot", "casino", "bet", "gamble",
        "discount", "sale", "price", "cheap", "expensive",
        "quality", "guarantee", "warranty", "refund", "return"
        ]
        self.vocab_map = {word: i for i, word in enumerate(self.vocabulary)}
        self.model = SVC(kernel='rbf', C=10, gamma=0.1)
        self.is_trained = False

    def get_model_size_kb(self) -> float:
        # Отримуємо основні параметри моделі
        model_params = {
            'support_vectors': self.model.support_vectors_.nbytes,
            'dual_coef': self.model.dual_coef_.nbytes,
            'intercept': self.model.intercept_.nbytes,
            'n_support': self.model.n_support_.nbytes,
            'support': self.model.support_.nbytes,
            'gamma': sys.getsizeof(self.model._gamma),
            'C': sys.getsizeof(self.model.C),
            'kernel': sys.getsizeof(self.model.kernel),
            'class_weight': sys.getsizeof(self.model.class_weight_),
            'classes': self.model.classes_.nbytes
        }
        
        # Рахуємо загальний розмір
        total_size = sum(model_params.values())
        
        # Додаємо розмір словника
        vocab_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.vocab_map.items())
        
        return (total_size + vocab_size) / 1024  # конвертуємо в КБ

    def extract_features(self, message: str) -> np.ndarray:
        sample = np.zeros(len(self.vocabulary))
        words = message.lower().split()
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word in self.vocab_map:
                sample[self.vocab_map[word]] += 1
        return sample

    def train(self, training_data: List[Tuple[str, int]]):
        if not training_data:
            raise RuntimeError("Training data is empty")
        
        X = np.array([self.extract_features(msg) for msg, _ in training_data])
        y = np.array([label for _, label in training_data])
        
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, message: str) -> int:
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        
        features = self.extract_features(message).reshape(1, -1)
        return int(self.model.predict(features)[0])

    def evaluate(self, test_data: List[Tuple[str, int]]) -> ClassificationMetrics:
        if not self.is_trained:
            raise RuntimeError("Model is not trained")

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for message, true_label in test_data:
            prediction = self.predict(message)
            
            if prediction == 1 and true_label == 1:
                true_positives += 1
            elif prediction == 1 and true_label == -1:
                false_positives += 1
            elif prediction == -1 and true_label == -1:
                true_negatives += 1
            elif prediction == -1 and true_label == 1:
                false_negatives += 1

        accuracy = (true_positives + true_negatives) / len(test_data)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return ClassificationMetrics(accuracy, precision, recall, f1_score)


class DataLoader:
    @staticmethod
    def load_from_csv(filename: str) -> List[Tuple[str, int]]:
        if not os.path.exists(filename):
            raise RuntimeError(f"Cannot open file: {filename}")

        try:
            df = pd.read_csv(filename, quoting=1)  # QUOTE_ALL mode
            data = [(str(row['message']), int(row['label'])) for _, row in df.iterrows()]
            print(f"Successfully loaded {len(data)} records from {filename}")
            return data
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")


def print_metrics(metrics: ClassificationMetrics):
    print("Classification Metrics:")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1_score:.4f}")


def main():
    try:
        # File paths
        train_file = "train_data.csv"
        test_file = "test_data.csv"

        # Print file information
        print("Checking files:")
        print(f"Train file path: {os.path.abspath(train_file)}")
        print(f"Test file path: {os.path.abspath(test_file)}")
        print(f"Train file exists: {os.path.exists(train_file)}")
        print(f"Test file exists: {os.path.exists(test_file)}")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise RuntimeError("Training or test data files not found")

        # Load data
        training_data = DataLoader.load_from_csv(train_file)
        test_data = DataLoader.load_from_csv(test_file)

        # Create and train classifier
        classifier = SpamClassifier()

        # Measure training time
        train_start = time.time()
        classifier.train(training_data)
        train_end = time.time()
        print(f"\nTraining time: {train_end - train_start:.6f} seconds")

        # Measure testing time
        test_start = time.time()
        metrics = classifier.evaluate(test_data)
        test_end = time.time()
        print(f"Testing time: {test_end - test_start:.6f} seconds")

        # Print model size
        model_size = classifier.get_model_size_kb()
        print(f"Model Size: {model_size:.2f} KB\n")

        print_metrics(metrics)

        # Test single message
        test_msg = "Click now to get free money"
        single_predict_start = time.time()
        prediction = classifier.predict(test_msg)
        single_predict_end = time.time()

        print(f"\nTest Message: \"{test_msg}\"")
        print(f"Prediction: {'SPAM' if prediction > 0 else 'NOT SPAM'}")
        print(f"Prediction time: {single_predict_end - single_predict_start:.2f} seconds")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()
