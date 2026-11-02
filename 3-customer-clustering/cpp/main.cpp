#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <sstream>


struct Customer {
    int operator_id;
    double call_minutes;
    double sms_count;
    double internet_mb;
    int last_activity;
    int age;
    int prepaid;
};


std::vector<Customer> readDataFromCSV(const std::string& filename) {
    std::vector<Customer> customers;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return customers;
    }

    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Customer c;

        std::getline(ss, value, ',');
        c.operator_id = std::stoi(value);

        std::getline(ss, value, ',');
        c.call_minutes = std::stod(value);

        std::getline(ss, value, ',');
        c.sms_count = std::stod(value);

        std::getline(ss, value, ',');
        c.internet_mb = std::stod(value);

        std::getline(ss, value, ',');
        c.last_activity = std::stoi(value);

        std::getline(ss, value, ',');
        c.age = std::stoi(value);

        std::getline(ss, value, ',');
        c.prepaid = std::stoi(value);

        customers.push_back(c);
    }
    return customers;
}


arma::mat convertToMatrix(const std::vector<Customer>& customers) {
    arma::mat data(7, customers.size());

    for (size_t i = 0; i < customers.size(); ++i) {
        data(0, i) = customers[i].operator_id;
        data(1, i) = customers[i].call_minutes;
        data(2, i) = customers[i].sms_count;
        data(3, i) = customers[i].internet_mb;
        data(4, i) = customers[i].last_activity;
        data(5, i) = customers[i].age;
        data(6, i) = customers[i].prepaid;
    }

    return data;
}

double euclideanDistance(const arma::vec& a, const arma::vec& b) {
    return arma::norm(a - b);
}

double calculateSilhouetteScore(const arma::mat& data, const arma::Row<size_t>& assignments, int k) {
    double totalScore = 0.0;
    int validPoints = 0;

    for (size_t i = 0; i < data.n_cols; ++i) {

        double a = 0.0;
        int clusterSize = 0;
        for (size_t j = 0; j < data.n_cols; ++j) {
            if (assignments(j) == assignments(i) && i != j) {
                a += euclideanDistance(data.col(i), data.col(j));
                clusterSize++;
            }
        }
        if (clusterSize > 0) {
            a /= clusterSize;
        }

        double b = std::numeric_limits<double>::max();
        for (int cluster = 0; cluster < k; ++cluster) {
            if (cluster != assignments(i)) {
                double clusterDist = 0.0;
                int otherClusterSize = 0;
                for (size_t j = 0; j < data.n_cols; ++j) {
                    if (assignments(j) == cluster) {
                        clusterDist += euclideanDistance(data.col(i), data.col(j));
                        otherClusterSize++;
                    }
                }
                if (otherClusterSize > 0) {
                    clusterDist /= otherClusterSize;
                    b = std::min(b, clusterDist);
                }
            }
        }

        if (b > a) {
            double s = (b - a) / std::max(a, b);
            totalScore += s;
            validPoints++;
        }
    }

    return validPoints > 0 ? totalScore / validPoints : 0.0;
}


double calculateIntraClusterDistance(const arma::mat& data, const arma::Row<size_t>& assignments, int k) {
    double totalDistance = 0.0;
    int totalPoints = 0;

    for (int cluster = 0; cluster < k; ++cluster) {
        std::vector<arma::vec> clusterPoints;
        for (size_t i = 0; i < data.n_cols; ++i) {
            if (assignments(i) == cluster) {
                clusterPoints.push_back(data.col(i));
            }
        }

        for (size_t i = 0; i < clusterPoints.size(); ++i) {
            for (size_t j = i + 1; j < clusterPoints.size(); ++j) {
                totalDistance += euclideanDistance(clusterPoints[i], clusterPoints[j]);
                totalPoints++;
            }
        }
    }

    return totalPoints > 0 ? totalDistance / totalPoints : 0.0;
}


void saveResults(const std::vector<Customer>& customers,
                const arma::Row<size_t>& assignments,
                const arma::mat& centroids,
                const std::string& filename) {
    std::ofstream file(filename);
    file << "operator_id,call_minutes,sms_count,internet_mb,last_activity,age,prepaid,cluster\n";

    for (size_t i = 0; i < customers.size(); ++i) {
        file << customers[i].operator_id << ","
             << customers[i].call_minutes << ","
             << customers[i].sms_count << ","
             << customers[i].internet_mb << ","
             << customers[i].last_activity << ","
             << customers[i].age << ","
             << customers[i].prepaid << ","
             << assignments[i] << "\n";
    }
    file.close();
}

int main() {
    // Phase 1: Training
    std::cout << "Phase 1: Training\n";
    std::vector<Customer> train_customers = readDataFromCSV("data/clients_mobile_train_data.csv");

    if (train_customers.empty()) {
        std::cerr << "No training data was read from the file. Exiting " << std::endl;
        return 1;
    }

    std::cout << "Read " << train_customers.size() << " customers from training file." << std::endl;

    arma::mat train_data = convertToMatrix(train_customers);
    
    // Calculate L2 norms for each column
    arma::vec column_norms(train_data.n_rows);
    for (size_t i = 0; i < train_data.n_rows; ++i) {
        column_norms(i) = arma::norm(train_data.row(i));
    }

    // Scale the training data
    arma::mat normalized_train_data = train_data;
    for (size_t i = 0; i < train_data.n_rows; ++i) {
        normalized_train_data.row(i) /= column_norms(i);
    }

    const int k = 3;
    mlpack::KMeans<> kmeans;

    auto train_start = std::chrono::high_resolution_clock::now();

    arma::Row<size_t> train_assignments;
    arma::mat centroids;
    kmeans.Cluster(normalized_train_data, k, train_assignments, centroids);

    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end - train_start;

    // Save training metrics
    double train_silhouette = calculateSilhouetteScore(normalized_train_data, train_assignments, k);
    double train_intra_cluster = calculateIntraClusterDistance(normalized_train_data, train_assignments, k);

    std::cout << "\nTraining Metrics:\n";
    std::cout << "Training time: " << train_duration.count() << " seconds\n";
    std::cout << "Silhouette Score: " << train_silhouette << "\n";
    std::cout << "Intra-cluster Distance: " << train_intra_cluster << "\n\n";

    // Phase 2: Testing
    std::cout << "Phase 2: Testing\n";
    std::vector<Customer> test_customers = readDataFromCSV("data/clients_mobile_test_data.csv");

    if (test_customers.empty()) {
        std::cerr << "No test data was read from the file. Exiting " << std::endl;
        return 1;
    }

    std::cout << "Read " << test_customers.size() << " customers from test file." << std::endl;

    arma::mat test_data = convertToMatrix(test_customers);

    // Scale test data using the same norms from training
    arma::mat normalized_test_data = test_data;
    for (size_t i = 0; i < test_data.n_rows; ++i) {
        normalized_test_data.row(i) /= column_norms(i);
    }

    // Assign test points to clusters
    auto test_start = std::chrono::high_resolution_clock::now();
    
    arma::Row<size_t> test_assignments(normalized_test_data.n_cols);
    for (size_t i = 0; i < normalized_test_data.n_cols; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        size_t closest_centroid = 0;

        for (size_t j = 0; j < centroids.n_cols; ++j) {
            double dist = arma::norm(normalized_test_data.col(i) - centroids.col(j));
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = j;
            }
        }
        test_assignments(i) = closest_centroid;
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_duration = test_end - test_start;

    // Calculate test metrics
    double test_silhouette = calculateSilhouetteScore(normalized_test_data, test_assignments, k);
    double test_intra_cluster = calculateIntraClusterDistance(normalized_test_data, test_assignments, k);

    std::cout << "\nTest Metrics:\n";
    std::cout << "Assignment time: " << test_duration.count() << " seconds\n";
    std::cout << "Silhouette Score: " << test_silhouette << "\n";
    std::cout << "Intra-cluster Distance: " << test_intra_cluster << "\n";

    // Calculate model size
    double model_size_kb = 0.0;
    
    // Size of centroids
    model_size_kb += (centroids.n_rows * centroids.n_cols * sizeof(double));
    
    // Size of column norms
    model_size_kb += (column_norms.n_elem * sizeof(double));
    
    // Convert to KB
    model_size_kb /= 1024.0;
    
    std::cout << "Model size: " << std::fixed << std::setprecision(2) << model_size_kb << " KB\n\n";

    // Save test results
    saveResults(test_customers, test_assignments, centroids, "data/clustering_results.csv");

    // Print cluster analysis for test data
    std::cout << "Test Data Cluster Analysis:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "\nCluster " << i << " Summary:\n";
        
        int clusterSize = 0;
        double avgCalls = 0, avgSms = 0, avgInternet = 0, avgAge = 0;

        for (size_t j = 0; j < test_customers.size(); ++j) {
            if (test_assignments(j) == i) {
                clusterSize++;
                avgCalls += test_customers[j].call_minutes;
                avgSms += test_customers[j].sms_count;
                avgInternet += test_customers[j].internet_mb;
                avgAge += test_customers[j].age;
            }
        }

        if (clusterSize > 0) {
            avgCalls /= clusterSize;
            avgSms /= clusterSize;
            avgInternet /= clusterSize;
            avgAge /= clusterSize;

            std::cout << "Size: " << clusterSize << " customers ("
                     << std::fixed << std::setprecision(1)
                     << (100.0 * clusterSize / test_customers.size()) << "%)\n";

            std::cout << "\nAverage Usage Patterns:\n";
            std::cout << "- Calls: " << std::fixed << std::setprecision(1) << avgCalls << " minutes/month\n";
            std::cout << "- SMS: " << std::fixed << std::setprecision(1) << avgSms << " messages/month\n";
            std::cout << "- Internet: " << std::fixed << std::setprecision(1) << avgInternet << " MB/month\n";
            std::cout << "- Average Age: " << std::fixed << std::setprecision(1) << avgAge << " years\n";

            std::cout << "\nCluster Profile: ";
            // Calculate relative usage patterns
            double internet_ratio = avgInternet / 5000.0;  // Normalize to 5000 MB
            double calls_ratio = avgCalls / 250.0;         // Normalize to 250 minutes
            double sms_ratio = avgSms / 50.0;             // Normalize to 50 SMS

            // Find the dominant usage pattern
            if (internet_ratio > calls_ratio && internet_ratio > sms_ratio && internet_ratio > 1.0) {
                std::cout << "Heavy Internet Users";
            } else if (calls_ratio > internet_ratio && calls_ratio > sms_ratio && calls_ratio > 1.0) {
                std::cout << "Heavy Call Users";
            } else if (sms_ratio > internet_ratio && sms_ratio > calls_ratio && sms_ratio > 1.0) {
                std::cout << "Heavy SMS Users";
            } else {
                std::cout << "Moderate Users";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
