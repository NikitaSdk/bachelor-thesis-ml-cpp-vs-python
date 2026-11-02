import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os

# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4) # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # Output layer
        return x

# Function to load MNIST data from CSV
def load_mnist_data_csv(data_path, labels_path):
    print("\n=== Loading data from files ===")
    print(f"Data file: {data_path}")
    print(f"Labels file: {labels_path}")

    try:
        data_df = pd.read_csv(data_path)
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        raise

    print("Files loaded successfully")

    # Convert pandas DataFrames to numpy arrays, then to PyTorch tensors
    # Data CSV contains 784 pixel values per row
    images = data_df.values.astype(np.float32)
    # Labels CSV contains 1 label per row
    labels = labels_df.values.flatten().astype(np.int64) # Use int64 for labels

    # Reshape images to (num_samples, channels, height, width)
    # Scale pixel values from 0-255 to 0-1 and add channel dimension
    images = torch.from_numpy(images).view(-1, 1, 28, 28) / 255.0
    labels = torch.from_numpy(labels)

    print(f"Successfully loaded {len(images)} images and {len(labels)} labels")
    if len(images) == 0:
        print("WARNING: No images were loaded!")

    if len(images) != len(labels):
        raise RuntimeError(f"Number of images ({len(images)}) does not match number of labels ({len(labels)})")

    return images, labels

# --- Main execution ---
if __name__ == "__main__":
    try:
        # Define data paths
        data_dir = 'C:/Users/user/PycharmProjects/GenDataForLabTest4C(CNN)/'
        train_data_path = os.path.join(data_dir, 'mnist_train.csv')
        train_labels_path = os.path.join(data_dir, 'mnist_train_labels.csv')
        test_data_path = os.path.join(data_dir, 'mnist_test.csv')
        test_labels_path = os.path.join(data_dir, 'mnist_test_labels.csv')

        print("Checking files...")
        print(f"Train data exists: {os.path.exists(train_data_path)}")
        print(f"Train labels exist: {os.path.exists(train_labels_path)}")
        print(f"Test data exists: {os.path.exists(test_data_path)}")
        print(f"Test labels exist: {os.path.exists(test_labels_path)}")

        # Load data
        training_images, training_labels = load_mnist_data_csv(train_data_path, train_labels_path)
        test_images, test_labels = load_mnist_data_csv(test_data_path, test_labels_path)

        print(f"Training data size: {len(training_images)}")
        print(f"Test data size: {len(test_images)}")

        # Create network, loss function, and optimizer
        net = Net()
        criterion = nn.CrossEntropyLoss()
        # Using Adam optimizer, can switch to SGD as in C++ if preferred, but Adam is often better
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        # Using a learning rate scheduler similar to set_min_learning_rate
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Reduce LR every 5 epochs by factor 0.1

        num_epochs = 15
        mini_batch_size = 128

        # Create and open CSV file for saving metrics
        metrics_file_path = 'training_metrics_python.csv'
        metrics_file = open(metrics_file_path, 'w')
        metrics_file.write("epoch,train_accuracy,train_loss,test_accuracy,test_loss\n")

        # --- Training ---
        print("\n=== Training Network ===")
        print(f"{'Epoch':<8}{'Train Accuracy':>15}{'Train Loss':>15}{'Test Accuracy':>15}{'Test Loss':>15}")
        print("-" * 70)

        training_start_time = time.time()

        for epoch in range(num_epochs):
            net.train() # Set model to training mode
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Create batches manually (similar to C++ logic)
            for i in range(0, len(training_images), mini_batch_size):
                batch_images = training_images[i:i + mini_batch_size]
                batch_labels = training_labels[i:i + mini_batch_size]

                # Forward pass
                outputs = net(batch_images)
                loss = criterion(outputs, batch_labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_labels.size(0)
                correct_train += (predicted == batch_labels).sum().item()

            scheduler.step() # Update learning rate

            # Calculate epoch metrics
            epoch_train_loss = running_loss / (len(training_images) / mini_batch_size)
            epoch_train_accuracy = correct_train / total_train

            # --- Evaluation (on test set) ---
            net.eval() # Set model to evaluation mode
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            with torch.no_grad(): # Disable gradient calculation for evaluation
                test_outputs = net(test_images)
                test_loss = criterion(test_outputs, test_labels).item()

                _, predicted_test = torch.max(test_outputs.data, 1)
                total_test += test_labels.size(0)
                correct_test += (predicted_test == test_labels).sum().item()

            epoch_test_accuracy = correct_test / total_test
            # Note: Calculating test loss based on criterion(test_outputs, test_labels) is standard
            # The C++ code used -log(test_accuracy) which is an approximation for CE loss close to 1 accuracy.
            # We will use the actual loss from the criterion for consistency.

            # Print metrics to console
            print(f"{epoch + 1:<8}{epoch_train_accuracy:>15.4f}{epoch_train_loss:>15.4f}{epoch_test_accuracy:>15.4f}{test_loss:>15.4f}")

            # Save metrics to CSV file
            metrics_file.write(f"{epoch + 1},{epoch_train_accuracy:.6f},{epoch_train_loss:.6f},{epoch_test_accuracy:.6f},{test_loss:.6f}\n")


        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        # Close the metrics file
        metrics_file.close()

        # --- Saving Model ---
        model_save_path = 'cnn_network_python.pth'
        print(f"\nSaving network to {model_save_path}...")
        torch.save(net.state_dict(), model_save_path)

        # Calculate model size
        model_size_bytes = os.path.getsize(model_save_path)
        model_size_kb = model_size_bytes / 1024.0

        # --- Final Testing ---
        print("\n=== Testing Network ===")
        print("Running final test on test dataset...")

        testing_start_time = time.time()

        net.eval() # Ensure model is in evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            final_test_outputs = net(test_images)
            _, predicted = torch.max(final_test_outputs.data, 1)
            total = test_labels.size(0)
            correct = (predicted == test_labels).sum().item()

        final_accuracy = correct / total

        testing_end_time = time.time()
        testing_duration = testing_end_time - testing_start_time # Duration in seconds

        print("\nTest Results:")
        print(f"  - Total test samples: {total}")
        print(f"  - Correct predictions: {correct}")
        print(f"  - Final accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")

        # Print timing and model size information
        print("\nPerformance Metrics:")
        print(f"  - Training time: {training_duration:.2f} seconds")
        print(f"  - Testing time: {testing_duration:.4f} seconds") # Testing time often < 1 second
        print(f"  - Model size: {model_size_kb:.2f} KB")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
