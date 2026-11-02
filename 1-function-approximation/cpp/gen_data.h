#ifndef GEN_DATA_H
#define GEN_DATA_H

#include <string>
#include <functional>

void generate_train_data(const std::string& filename, int num_samples, float x_min, float x_max, std::function<float(float)> target_function);

#endif
