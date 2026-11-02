#include "SpamClassifier.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>

SpamClassifier::SpamClassifier() : isTrained(false) {
    vocabulary = {
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
    };
    
    trainer.set_c(10);


    trainer.set_kernel(kernel_type(0.1));
}

SpamClassifier::sample_type SpamClassifier::extractFeatures(const std::string& message) {
    sample_type sample;
    sample = 0;
    std::string word;
    std::istringstream iss(message);
    std::map<std::string, int> vocab_map;

    for (size_t i = 0; i < vocabulary.size(); ++i)
        vocab_map[vocabulary[i]] = i;

    while (iss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        if (vocab_map.count(word) > 0)
            sample(vocab_map[word]) += 1;
    }

    return sample;
}

void SpamClassifier::train(const std::vector<std::pair<std::string, int>>& trainingData) {
    if (trainingData.empty()) {
        throw std::runtime_error("Training data is empty");
    }

    std::vector<sample_type> samples;
    std::vector<double> labels;

    for (const auto& [message, label] : trainingData) {
        samples.push_back(extractFeatures(message));
        labels.push_back(label);
    }

    model = trainer.train(samples, labels);
    isTrained = true;
}

int SpamClassifier::predict(const std::string& message) {
    if (!isTrained) {
        throw std::runtime_error("Model is not trained");
    }
    return model(extractFeatures(message)) > 0 ? 1 : -1;
}

ClassificationMetrics SpamClassifier::evaluate(const std::vector<std::pair<std::string, int>>& testData) {
    if (!isTrained) {
        throw std::runtime_error("Model is not trained");
    }

    int truePositives = 0;
    int falsePositives = 0;
    int trueNegatives = 0;
    int falseNegatives = 0;

    for (const auto& [message, trueLabel] : testData) {
        int prediction = predict(message);
        
        if (prediction == 1 && trueLabel == 1) truePositives++;
        else if (prediction == 1 && trueLabel == -1) falsePositives++;
        else if (prediction == -1 && trueLabel == -1) trueNegatives++;
        else if (prediction == -1 && trueLabel == 1) falseNegatives++;
    }

    ClassificationMetrics metrics;
    metrics.accuracy = (double)(truePositives + trueNegatives) / testData.size();
    metrics.precision = truePositives / (double)(truePositives + falsePositives);
    metrics.recall = truePositives / (double)(truePositives + falseNegatives);
    metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);

    return metrics;
}

ModelInfo SpamClassifier::getModelInfo() const {
    ModelInfo info;
    
    // Базова інформація
    info.vocabularySize = vocabulary.size();
    info.featureVectorSize = 50; // Оновлено з 10 на 50

    if (!isTrained) {
        info.supportVectorsCount = 0;
        info.modelMemorySize = 0;
        return info;
    }

    // Отримуємо кількість опорних векторів
    info.supportVectorsCount = model.basis_vectors.size();
    
    // Розрахунок розміру моделі в байтах
    double modelSize = 0;
    
    // Розмір опорних векторів
    modelSize += info.supportVectorsCount * info.featureVectorSize * sizeof(double);
    
    // Розмір альфа коефіцієнтів
    modelSize += info.supportVectorsCount * sizeof(double);
    
    // Розмір словника
    modelSize += vocabulary.size() * sizeof(std::string);
    
    // Додаткові параметри (kernel, bias, тощо)
    modelSize += sizeof(kernel_type) + sizeof(double);
    
    // Конвертуємо в кілобайти
    info.modelMemorySize = modelSize / 1024.0;
    
    return info;
} 
