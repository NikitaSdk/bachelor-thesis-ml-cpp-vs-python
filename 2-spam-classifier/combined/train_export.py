import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import os

def create_spam_vocabulary():
    spam_words = {
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
    }
    # Create word to index mapping
    word_index = {word: idx + 1 for idx, word in enumerate(sorted(spam_words))}
    return word_index

def tokenize_text(text, word_index):
    words = text.lower().split()
    return [word_index.get(word, 0) for word in words]

def train_and_save_model():
    # Print file path and check existence
    train_file_path = os.path.abspath('train_data.csv')
    print("Checking files:")
    print(f"Train file path: {train_file_path}")
    print(f"Train file exists: {int(os.path.exists(train_file_path))}")

    # Create vocabulary
    word_index = create_spam_vocabulary()
    vocab_size = len(word_index) + 1  # +1 for padding/unknown words

    # Load and preprocess data
    df = pd.read_csv('train_data.csv')
    print(f"Successfully loaded {len(df)} records from {train_file_path}")

    # Convert labels from -1/1 to 0/1
    df['label'] = (df['label'] == 1).astype(int)

    # Tokenization and padding
    max_length = 50
    sequences = [tokenize_text(text, word_index) for text in df['message']]
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Convert to float32
    padded_sequences = padded_sequences.astype(np.float32)

    # Split the data for validation during training
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, 
        df['label'], 
        test_size=0.2, 
        random_state=42
    )

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,), dtype=tf.float32),
        tf.keras.layers.Embedding(vocab_size, 32),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train,
              epochs=10,
              validation_data=(X_val, y_val),
              batch_size=32,
              verbose=1)
    training_time = time.time() - start_time

    # Test the model on some examples
    test_texts = [
        "Win a free iPhone now! Click here",
        "Meeting tomorrow at 2 PM",
        "URGENT: Your account needs verification",
        "Please review the project proposal"
    ]
    
    # Tokenize and pad test texts
    test_sequences = [tokenize_text(text, word_index) for text in test_texts]
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    test_padded = test_padded.astype(np.float32)  # Convert to float32
    
    print("\nТестування на Python:")
    predictions = model.predict(test_padded)
    for text, pred in zip(test_texts, predictions):
        result = "SPAM" if pred[0] > 0.5 else "NOT SPAM"
        confidence = float(pred[0] if pred[0] > 0.5 else 1 - pred[0])
        print(f"Text: {text}")
        print(f"Prediction: {result} (впевненість: {confidence:.2f})")

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open('spam_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    print("\nМодель збережена як 'spam_classifier.tflite'")

    # Test TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nТестування TensorFlow Lite моделі:")
    for text, sequence in zip(test_texts, test_padded):
        interpreter.set_tensor(input_details[0]['index'], [sequence])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        result = "SPAM" if output > 0.5 else "NOT SPAM"
        confidence = float(output) if output > 0.5 else float(1 - output)
        print(f"Text: {text}")
        print(f"Prediction: {result} (впевненість: {confidence:.2f})")

    # Save vocabulary with header
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        f.write(f"# Total words: {len(word_index)}\n")
        f.write("# Format: word index\n")
        f.write("#------------------\n")
        for word, index in sorted(word_index.items(), key=lambda x: x[1]):
            f.write(f"{word} {index}\n")

    print(f"\nTraining time: {training_time:.6f} seconds")
    model_size = os.path.getsize('spam_classifier.tflite') / 1024
    print(f"Model Size: {model_size:.2f} KB")

if __name__ == "__main__":
    train_and_save_model()
