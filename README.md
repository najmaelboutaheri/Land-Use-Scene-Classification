# Land Use Scene Classification

This repository contains the Jupyter notebook for **Land Use Scene Classification**. The notebook includes the process of data loading, preprocessing, model training, and evaluation for classifying land use scenes using various machine learning or deep learning techniques, specifically Convolutional Neural Networks (CNN).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to classify land use scenes using image classification techniques. The dataset contains images from different land use categories, and the notebook walks through the entire process of building a classification model from scratch, including:

- Data loading and exploration
- Image preprocessing and augmentation
- Model selection and training using CNN architectures
- Evaluation of the model on a test set

## Dataset

The dataset used in this project consists of various land use scene images, categorized into different classes. You can download the dataset from [this link](#).

**Categories** include:

- Urban
- Agriculture
- Forest
- Water bodies
- And more...

Each image in the dataset represents a scene, and the model predicts the type of land use.

## Dependencies

To run the notebook, the following dependencies are required:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- Python 3.x
- Jupyter Notebook
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Matplotlib
- Pandas
- NumPy

# Usage

1. Clone this repository:
   \```bash
   git clone https://github.com/your-repo/land-use-scene-classification.git
   \```
2. Navigate to the directory and open the notebook:
   \```bash
   cd land-use-scene-classification
   jupyter notebook land-use-scene-classification-all-process.ipynb
   \```
3. Run the notebook cells step-by-step to reproduce the results.
   
## Model Architecture

This project uses two CNN architectures for classifying land use scenes. The first CNN architecture is more complex, while the second uses separable convolutions to reduce computational complexity.
### CNN Architecture
 ```bash
Model = Sequential()

Model.add(SeparableConv2D(32, 3, activation="relu", input_shape=(226, 226, 3)))
Model.add(BatchNormalization())
Model.add(MaxPooling2D((2)))

Model.add(SeparableConv2D(64, 3, activation="relu"))
Model.add(SeparableConv2D(128, (3,3), activation="relu"))
Model.add(Dropout(0.5))
Model.add(MaxPooling2D((2)))

Model.add(SeparableConv2D(128, 3, activation="relu"))
Model.add(SeparableConv2D(128, 3, activation="relu"))
Model.add(Dropout(0.5))
Model.add(GlobalAveragePooling2D())

Model.add(Flatten())
Model.add(Dense(256, activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(21, activation="softmax")) 
```
## Results

The classification model achieves the following metrics:

- **Accuracy**: `80%`
- **Loss**: `0.69`


You can view the results and evaluation metrics in detail inside the notebook.

## Contributing

Contributions are welcome! If you find any issues or want to add enhancements, feel free to create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
