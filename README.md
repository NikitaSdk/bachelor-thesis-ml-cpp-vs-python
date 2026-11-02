# ML on C++ vs Python — Bachelor's Thesis

**Bachelor's thesis, Igor Sikorsky Kyiv Polytechnic Institute (2025), specialty 122 Computer Science.**
Topic: *Libraries for implementing machine learning algorithms in C++.*

Four machine-learning tasks, each solved three ways — pure **C++**, pure **Python**, and a **combined** approach (train in Python → export the model → run inference in C++) — to compare accuracy, speed and ease of development.

![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TF_Lite-FF6F00?logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Tasks

| # | Task | C++ | Python | Combined (Python → C++) |
| --- | --- | --- | --- | --- |
| 01 | [Function approximation](01-function-approximation) | FANN | PyTorch | PyTorch → ONNX → ONNX Runtime |
| 02 | [Spam classification](02-spam-classifier) | Dlib (SVM) | scikit-learn | TensorFlow → TF Lite C++ API |
| 03 | [Customer clustering](03-customer-clustering) | mlpack (k-means) | scikit-learn | TensorFlow → TF Lite C++ API |
| 04 | [Handwritten digit recognition (CNN)](04-digit-recognition-cnn) | Dlib DNN | PyTorch | PyTorch → ONNX → OpenCV DNN |

## Structure

```
NN-task/
├── cpp/                  pure C++ implementation
├── python/               pure Python implementation
└── combined/
    ├── train_export.py   training in Python and model export
    └── cpp_inference/    inference of the exported model in C++
```

Datasets are read from a local `data/` folder. The clustering dataset is included; MNIST (task 04) is the standard dataset exported to CSV.

## Tools

C++17, Visual Studio / CLion, PyCharm, Windows. Libraries: FANN, Dlib, mlpack, ONNX Runtime, TensorFlow Lite C++ API, OpenCV; PyTorch, TensorFlow, scikit-learn, pandas, matplotlib.
