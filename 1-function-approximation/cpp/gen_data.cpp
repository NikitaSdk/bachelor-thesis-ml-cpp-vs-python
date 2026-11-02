#include "gen_data.h"
#include <fstream>
#include <iostream>

void generate_train_data(const std::string& filename, int num_samples, float x_min, float x_max, std::function<float(float)> target_function)
{
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Open file error" << std::endl;
        return;
    }

    fout << num_samples << " 1 1" << std::endl;

    float step = (x_max - x_min) / (num_samples - 1);
    for (int i = 0; i < num_samples; ++i) {
        float x = x_min + i * step;
        float y = target_function(x);
        fout << x << std::endl;
        fout << y << std::endl;
    }

    fout.close();
    std::cout << "File " << filename << " created" << std::endl;
}
