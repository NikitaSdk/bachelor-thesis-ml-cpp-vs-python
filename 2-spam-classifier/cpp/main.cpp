#include "SpamClassifier.h"
#include "DataLoader.h"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

void printMetrics(const ClassificationMetrics& metrics) {
    cout << fixed << setprecision(4);
    cout << "Classification Metrics:" << endl;
    cout << "Accuracy:  " << metrics.accuracy << endl;
    cout << "Precision: " << metrics.precision << endl;
    cout << "Recall:    " << metrics.recall << endl;
    cout << "F1 Score:  " << metrics.f1_score << endl;
}

void printModelSize(const ModelInfo& info) {
    cout << fixed << setprecision(2);
    cout << "Model Size: " << info.modelMemorySize << " KB" << endl;
}

int main() {
    try {

        string trainFile = "data/train_data.csv";
        string testFile = "data/test_data.csv";


        cout << "Checking files:" << endl;
        cout << "Train file path: " << trainFile << endl;
        cout << "Test file path: " << testFile << endl;
        cout << "Train file exists: " << filesystem::exists(trainFile) << endl;
        cout << "Test file exists: " << filesystem::exists(testFile) << endl;

        if (!filesystem::exists(trainFile) || !filesystem::exists(testFile)) {
            throw runtime_error("Training or test data files not found");
        }

        auto trainingData = DataLoader::loadFromCSV(trainFile);
        auto testData = DataLoader::loadFromCSV(testFile);


        SpamClassifier classifier;

        auto trainStart = high_resolution_clock::now();
        classifier.train(trainingData);
        auto trainEnd = high_resolution_clock::now();
        duration<double> trainDuration = trainEnd - trainStart;
        cout << "\nTraining time: " << fixed << setprecision(6) << trainDuration.count() << " seconds" << endl;


        auto testStart = high_resolution_clock::now();
        auto metrics = classifier.evaluate(testData);
        auto testEnd = high_resolution_clock::now();
        duration<double> testDuration = testEnd - testStart;
        cout << "Testing time: " << fixed << setprecision(6) << testDuration.count() << " seconds" << endl;

        printModelSize(classifier.getModelInfo());
        cout << endl;

        printMetrics(metrics);

        string testMsg = "Click now to get free money";
        auto singlePredictStart = high_resolution_clock::now();
        int prediction = classifier.predict(testMsg);
        auto singlePredictEnd = high_resolution_clock::now();
        duration<double> singlePredictDuration = singlePredictEnd - singlePredictStart;

        cout << "\nTest Message: \"" << testMsg << "\"" << endl;
        cout << "Prediction: " << (prediction > 0 ? "SPAM" : "NOT SPAM") << endl;
        cout << "Prediction time: " << fixed << setprecision(2) << singlePredictDuration.count() << " seconds" << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
