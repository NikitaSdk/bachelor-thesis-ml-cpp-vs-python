#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>
#include <utility>

class DataLoader {
public:
    static std::vector<std::pair<std::string, int>> loadFromCSV(const std::string& filename);
};

#endif
