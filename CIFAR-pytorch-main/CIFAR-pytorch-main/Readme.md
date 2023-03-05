# CIFAR10 Image Classification with PyTorch

This is a deep learning project that uses PyTorch to build and train a convolutional neural network (CNN) for image classification on the CIFAR10 dataset. (:man_technologist:)

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Background

The CIFAR10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are mutually exclusive and correspond to airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is split into 50,000 training images and 10,000 test images.

The goal of this project is to build a CNN that can accurately classify images from the CIFAR10 dataset. We use PyTorch, a popular deep learning library, to build and train our model.

- This Dataset has 50K trainig samples and 10k test images for 10 different classes.
- The Data looks like this
![image](https://user-images.githubusercontent.com/40626584/222936510-c8d7b86b-daf1-4777-89d0-1ea5cfcbba15.png)

## Installation

To run this project, you'll need to install the following dependencies:

- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- Torchvision

The Folder structure for this repo is:

    ├── src                    # Source files
    │   ├── data_processing        # Load and stress tests
    │   ├── models     
    │   └── loss
    |   
    ├── main.py
    └── test.py
    
 ## Usage
 To train the model, you can run the train.py script. This will train the CNN on the CIFAR10 dataset and save the trained model as a .pth file.
 ```
 python main.py
 ```
 To test the trained model on the test set, you can run the test.py script. This will load the trained model and test it on the test set.
 ```
 python test.py
 ```

## Results

The Model Return the following accuracy results:
    ![image](https://user-images.githubusercontent.com/40626584/222936746-6d987e2c-d0e8-447f-a22f-6479cadc7b8b.png)

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
