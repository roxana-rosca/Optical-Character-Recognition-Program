# Optical Character Recognition using the K-Nearest Neighbors Algorithm (From Scratch)

This project demonstrates Optical Character Recognition (OCR) using the k-nearest neighbors algorithm implemented from scratch. It distinguishes between handwritten digits and clothing items with approximately 93% accuracy for digits and 77% accuracy for clothing items.

## Description

- This OCR project is developed in Python entirely from scratch without using any external libraries.
- It performs image recognition on the MNIST and Fashion-MNIST datasets using the k-nearest neighbors algorithm.
- The implementation achieves approximately 93% accuracy for recognizing handwritten digits and 77% for recognizing clothing items.
- Key points about the project:
  - No imports are used; everything is implemented from scratch.
  - It distinguishes between digits and clothes.
  - PIL and NumPy are imported for visualization and loading images, although they are not required for the code to function.
  - The K-NN algorithm assumes similarity between new data and available data points and classifies the new data based on its proximity to existing categories.
  - K-NN is known as a "lazy learner" algorithm because it does not learn from the training set immediately; instead, it stores the dataset and performs actions on it during classification.

## Credits
- **Tutorial**: [clumsy computer's Tutorial](https://www.youtube.com/watch?v=vzabeKdW9tE)
- **Dataset (Numbers)**: [MNIST Dataset](https://github.com/MrHeadbang/machineLearning/blob/main/mnist.zip)
- **Dataset (Fashion)**: [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
