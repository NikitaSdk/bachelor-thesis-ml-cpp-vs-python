#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::vector<std::pair<std::string, int>> DataLoader::loadFromCSV(const std::string& filename) {
    std::vector<std::pair<std::string, int>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("File is empty or cannot read header");
    }
    
    int lineNumber = 1;
    while (std::getline(file, line)) {
        lineNumber++;
        try {

            size_t lastComma = line.find_last_of(',');
            if (lastComma == std::string::npos) {
                throw std::runtime_error("No coma found in line");
            }


            std::string message = line.substr(0, lastComma);
            std::string label = line.substr(lastComma + 1);


            message.erase(std::remove(message.begin(), message.end(), '"'), message.end());
            label.erase(std::remove(label.begin(), label.end(), '"'), label.end());
            

            message.erase(0, message.find_first_not_of(" \t\r\n"));
            message.erase(message.find_last_not_of(" \t\r\n") + 1);
            label.erase(0, label.find_first_not_of(" \t\r\n"));
            label.erase(label.find_last_not_of(" \t\r\n") + 1);


            if (label != "1" && label != "-1") {
                throw std::runtime_error("Invalid label value: " + label);
            }

            int labelValue = std::stoi(label);
            data.push_back({message, labelValue});
        } catch (const std::exception& e) {
            std::cerr << "Error processing line " << lineNumber << ": " << e.what() << std::endl;
            std::cerr << "Line content: " << line << std::endl;
            throw;
        }
    }
    
    if (data.empty()) {
        throw std::runtime_error("No valid data found in file");
    }

    std::cout << "Successfully loaded " << data.size() << " records from " << filename << std::endl;
    return data;
}
