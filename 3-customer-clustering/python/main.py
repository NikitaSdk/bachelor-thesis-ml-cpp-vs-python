import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import sys
import os
import matplotlib.pyplot as plt


class Customer:
    def __init__(self, operator_id, call_minutes, sms_count, internet_mb, last_activity, age, prepaid):
        self.operator_id = operator_id
        self.call_minutes = call_minutes
        self.sms_count = sms_count
        self.internet_mb = internet_mb
        self.last_activity = last_activity
        self.age = age
        self.prepaid = prepaid


class KMeansTrainer:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = None
        self.training_time = 0
        self.assignment_time = 0
        self.normalized_data = None
        
    def normalize_data(self, data):
        """Нормалізація даних (як в C++ версії armadillo::normalise)"""
        normalized = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            col = data[:, i]
            col_norm = np.linalg.norm(col)
            if col_norm > 0:
                normalized[:, i] = col / col_norm
        return normalized
        
    def train(self, data):
        """Тренування моделі"""
        data_t = data.T
        self.normalized_data = self.normalize_data(data_t).T
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=1,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        
        train_start = time.time()
        self.model.fit(self.normalized_data.T)
        train_end = time.time()
        self.training_time = train_end - train_start
        
        assignments = self.model.predict(self.normalized_data.T)
        return assignments
        
    def test(self, data):
        """Тестування моделі на нових даних"""
        data_t = data.T
        normalized_data = self.normalize_data(data_t).T
        
        assign_start = time.time()
        assignments = self.model.predict(normalized_data.T)
        assign_end = time.time()
        self.assignment_time = assign_end - assign_start
        
        return assignments, normalized_data
        
    def get_model_size(self):
        """Розрахунок розміру моделі в KB"""
        if self.model is None:
            return 0
            
        model_size_bytes = 0
        model_size_bytes += (self.model.cluster_centers_.shape[0] * 
                           self.model.cluster_centers_.shape[1] * 8)
        model_size_bytes += (len(self.model.labels_) * 8)
        
        if self.normalized_data is not None:
            model_size_bytes += (2 * self.normalized_data.shape[0] * 8)
        
        return model_size_bytes / 1024.0


def read_data_from_csv(filename):
    try:
        df = pd.read_csv(filename)
        customers = []
        for _, row in df.iterrows():
            customer = Customer(
                operator_id=row['Operator'],
                call_minutes=row['CallMinutes'],
                sms_count=row['SMSCount'],
                internet_mb=row['InternetMB'],
                last_activity=row['LastActivityDays'],
                age=row['CustomerAge'],
                prepaid=row['Prepaid']
            )
            customers.append(customer)
        return customers
    except Exception as e:
        print(f"Error: Could not open file {filename}")
        print(f"Error details: {e}")
        return []


def convert_to_matrix(customers):
    data = np.zeros((len(customers), 7))
    for i, customer in enumerate(customers):
        data[i] = [
            customer.operator_id,
            customer.call_minutes,
            customer.sms_count,
            customer.internet_mb,
            customer.last_activity,
            customer.age,
            customer.prepaid
        ]
    return data.T


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_silhouette_score(data, assignments, k):
    total_score = 0.0
    valid_points = 0

    for i in range(data.shape[1]):
        a = 0.0
        cluster_size = 0
        for j in range(data.shape[1]):
            if assignments[j] == assignments[i] and i != j:
                a += euclidean_distance(data[:, i], data[:, j])
                cluster_size += 1
        
        if cluster_size > 0:
            a /= cluster_size

        b = float('inf')
        for cluster in range(k):
            if cluster != assignments[i]:
                cluster_dist = 0.0
                other_cluster_size = 0
                for j in range(data.shape[1]):
                    if assignments[j] == cluster:
                        cluster_dist += euclidean_distance(data[:, i], data[:, j])
                        other_cluster_size += 1
                
                if other_cluster_size > 0:
                    cluster_dist /= other_cluster_size
                    b = min(b, cluster_dist)

        if b > a and cluster_size > 0:
            s = (b - a) / max(a, b)
            total_score += s
            valid_points += 1

    return total_score / valid_points if valid_points > 0 else 0.0


def calculate_intra_cluster_distance(data, assignments, k):
    total_distance = 0.0
    total_points = 0

    for cluster in range(k):
        cluster_points = data[:, assignments == cluster]
        
        for i in range(cluster_points.shape[1]):
            for j in range(i + 1, cluster_points.shape[1]):
                total_distance += euclidean_distance(cluster_points[:, i], cluster_points[:, j])
                total_points += 1

    return total_distance / total_points if total_points > 0 else 0.0


def print_cluster_analysis(customers, assignments, normalized_data, n_clusters):
    print("\nTest Data Cluster Analysis:\n")
    
    for i in range(n_clusters):
        cluster_mask = assignments == i
        cluster_size = np.sum(cluster_mask)

        print(f"Cluster {i} Summary:")
        print("----------------------------------------")
        print(f"Size: {cluster_size} customers ({100.0 * cluster_size / len(customers):.1f}%)")

        if cluster_size > 0:
            avg_calls = np.mean([c.call_minutes for c, m in zip(customers, cluster_mask) if m])
            avg_sms = np.mean([c.sms_count for c, m in zip(customers, cluster_mask) if m])
            avg_internet = np.mean([c.internet_mb for c, m in zip(customers, cluster_mask) if m])
            avg_age = np.mean([c.age for c, m in zip(customers, cluster_mask) if m])

            print("\nAverage Usage Patterns:")
            print(f"- Calls: {avg_calls:.1f} minutes/month")
            print(f"- SMS: {avg_sms:.1f} messages/month")
            print(f"- Internet: {avg_internet:.1f} MB/month")
            print(f"- Average Age: {avg_age:.1f} years")

            print("\nCluster Profile:", end=" ")
            if avg_calls > 250 and avg_calls > avg_internet/20:
                print("Heavy Call Users")
            elif avg_internet > 5000 and avg_internet/20 > avg_calls:
                print("Heavy Internet Users")
            else:
                print("Moderate Users")
        print("----------------------------------------\n")


def main():
    np.random.seed(42)
    
    # Phase 1: Training
    print("Phase 1: Training")
    train_customers = read_data_from_csv("clients_mobile_train_data.csv")
    if not train_customers:
        print("No training data was read from the file. Exiting")
        return

    print(f"Read {len(train_customers)} customers from training file.\n")
    train_data = convert_to_matrix(train_customers)

    trainer = KMeansTrainer(n_clusters=3)
    train_assignments = trainer.train(train_data)
    
    train_silhouette = calculate_silhouette_score(trainer.normalized_data, train_assignments, trainer.n_clusters)
    train_intra_dist = calculate_intra_cluster_distance(trainer.normalized_data, train_assignments, trainer.n_clusters)

    print("Training Metrics:")
    print(f"Training time: {trainer.training_time:.7f} seconds")
    print(f"Silhouette Score: {train_silhouette:.6f}")
    print(f"Intra-cluster Distance: {train_intra_dist:.7f}\n")

    # Phase 2: Testing
    print("Phase 2: Testing")
    test_customers = read_data_from_csv("clients_mobile_test_data.csv")
    if not test_customers:
        print("No test data was read from the file. Exiting")
        return

    print(f"Read {len(test_customers)} customers from test file.\n")
    test_data = convert_to_matrix(test_customers)

    test_assignments, test_normalized_data = trainer.test(test_data)
    
    test_silhouette = calculate_silhouette_score(test_normalized_data, test_assignments, trainer.n_clusters)
    test_intra_dist = calculate_intra_cluster_distance(test_normalized_data, test_assignments, trainer.n_clusters)

    print("Test Metrics:")
    print(f"Assignment time: {trainer.assignment_time:.2e} seconds")
    print(f"Silhouette Score: {test_silhouette:.6f}")
    print(f"Intra-cluster Distance: {test_intra_dist:.7f}")
    print(f"Model size: {trainer.get_model_size():.2f} KB\n")

    print_cluster_analysis(test_customers, test_assignments, test_normalized_data, trainer.n_clusters)

    # Create DataFrame for visualization
    df = pd.DataFrame({
        'operator_id': [c.operator_id for c in test_customers],
        'call_minutes': [c.call_minutes for c in test_customers],
        'sms_count': [c.sms_count for c in test_customers],
        'internet_mb': [c.internet_mb for c in test_customers],
        'last_activity': [c.last_activity for c in test_customers],
        'age': [c.age for c in test_customers],
        'prepaid': [c.prepaid for c in test_customers],
        'cluster': test_assignments
    })

    # Scatter plot
    plt.figure(figsize=(12, 8))
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['call_minutes'],
                   cluster_data['internet_mb'],
                   label=f'Cluster {cluster}')
    plt.title('Кластеризація клієнтів')
    plt.xlabel('Хвилини розмови')
    plt.ylabel('Використання інтернету (MB)')
    plt.legend()
    plt.show()

    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Розподіл показників')

    axes[0,0].hist(df['call_minutes'], bins=20)
    axes[0,0].set_title('Розподіл хвилин розмови')
    axes[0,0].set_xlabel('Хвилини розмови')
    axes[0,0].set_ylabel('Кількість клієнтів')

    axes[0,1].hist(df['sms_count'], bins=20)
    axes[0,1].set_title('Розподіл SMS')
    axes[0,1].set_xlabel('Кількість SMS')
    axes[0,1].set_ylabel('Кількість клієнтів')

    axes[1,0].hist(df['internet_mb'], bins=20)
    axes[1,0].set_title('Розподіл використання інтернету')
    axes[1,0].set_xlabel('Використання інтернету (MB)')
    axes[1,0].set_ylabel('Кількість клієнтів')

    axes[1,1].hist(df['age'], bins=20)
    axes[1,1].set_title('Розподіл віку')
    axes[1,1].set_xlabel('Вік')
    axes[1,1].set_ylabel('Кількість клієнтів')

    plt.tight_layout()
    plt.show()

    # Cluster statistics
    print("\nСтатистика по кластерах:")
    cluster_stats = df.groupby('cluster')[['call_minutes', 'sms_count', 'internet_mb', 'age']].mean()
    print(cluster_stats)


if __name__ == "__main__":
    main()
