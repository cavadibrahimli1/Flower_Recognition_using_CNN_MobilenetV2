![image](https://github.com/cavadibrahimli1/Flower_Recognition_using_CNN_MobilenetV2/assets/76445357/e6793793-556d-4450-b05d-354298ad091b)



# Flower Recognition Using Convolutional Neural Network

## Introduction

This project focuses on using Convolutional Neural Networks (CNNs) to recognize and classify different species of flowers. The main objective is to create a robust and accurate model that can differentiate between various types of flowers, which has significant implications in agriculture and marketing. This repository contains the code and documentation for the flower recognition project.

## Abstract

Artificial intelligence is rapidly evolving, opening new avenues in various industries. Convolutional Neural Networks (CNNs) have proven to be highly effective in image recognition tasks. This project applies CNNs to flower recognition, demonstrating its potential in agriculture and marketing.

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Our Work](#our-work)
   - [Motivation](#motivation)
   - [Dataset](#dataset)
   - [Data Preparation](#data-preparation)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Creating the Model](#creating-the-model)
   - [Prediction](#prediction)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Challenges](#challenges)
6. [Future Work](#future-work)
7. [Conclusion](#conclusion)
8. [References](#references)

## Literature Review

### Key Studies

1. **Hanafiah et al. (2022)**: Deep CNN models using transfer learning for flower recognition.
2. **Mete and Ensari (2019)**: Classification systems integrating CNNs with various classifiers.
3. **Yifei et al. (2022)**: Enhanced CNN architectures for flower classification.
4. **Rajkomar and Pudaruth (2023)**: Integration of deep CNNs with traditional machine learning algorithms.

## Our Work

### Motivation

The detection of flowers using CNNs is critical for advancing agricultural practices and improving marketing strategies. CNN-based flower detection can optimize crop monitoring, targeted weed control, and harvesting scheduling, reducing manual labor and increasing efficiency.

### Dataset

The dataset used contains 4242 images of five flower types: tulip, daisy, dandelion, rose, and sunflower. It is sourced from Kaggle and includes diverse images to ensure robust model training.

![image](https://github.com/cavadibrahimli1/Flower_Recognition_using_CNN_MobilenetV2/assets/76445357/3d4bb3a2-7c5f-4cff-9526-5e5c92bb27e2)


### Data Preparation

1. **Loading Dataset**: Images and labels are loaded and organized into a Pandas DataFrame.
2. **Data Augmentation**: Techniques such as random rotation, shifting, shearing, zooming, and horizontal flipping are applied.
3. **Splitting Data**: The dataset is split into training and validation sets using a stratified approach.

### Exploratory Data Analysis

EDA involves analyzing class distribution and sample images to understand the dataset's balance and diversity. Visual inspections and pie charts are used to ensure a comprehensive understanding.

### Creating the Model

1. **Data Preparation**: The dataset is partitioned into training and test sets with augmentation applied to the training data.
2. **Model Compilation**: The MobileNetV2 model is fine-tuned and compiled using the Adam optimizer and categorical cross-entropy loss function.
3. **Training Process**: The model is trained with early stopping to prevent overfitting, using a batch size of 32 over 30 epochs.
4. **Evaluation**: Metrics such as accuracy, loss, confusion matrices, and classification reports are generated.

### Prediction

The prediction phase evaluates the model's effectiveness in classifying new images. Visualization of classification results and processing stages is provided to demonstrate the model's performance.

## Hyperparameter Tuning

### Data Augmentation

Data augmentation significantly improves model performance by artificially expanding the dataset with random transformations.

### Loss Function

The impact of different loss functions (Categorical Cross-Entropy, Mean Square Error, Sparse Categorical Cross-Entropy) is evaluated. Categorical Cross-Entropy shows the best results.

## Challenges

### Data Collection and Cleaning

Collecting and preparing high-quality data is labor-intensive and critical for model performance. Ensuring diverse and well-labeled images is essential.

### Model Implementation

Implementing CNNs requires high computational power and careful optimization of hyperparameters to prevent overfitting.

![image](https://github.com/cavadibrahimli1/Flower_Recognition_using_CNN_MobilenetV2/assets/76445357/a11ffbe3-6b94-4401-9103-f03448b0c937)


### Enhancing Model Accuracy

Techniques such as data augmentation, transfer learning, and ensemble methods are used to improve model accuracy and robustness.

## Future Work

Future research will focus on expanding the number of flower types, integrating flower detection systems into agricultural machinery, and enhancing computational efficiency for deployment in real-world applications.
![image](https://github.com/cavadibrahimli1/Flower_Recognition_using_CNN_MobilenetV2/assets/76445357/622e4a15-2e19-4e2b-9ffe-d8a2bff2446c)


## Conclusion

CNN-based flower recognition has the potential to revolutionize agriculture and marketing by providing accurate and efficient identification and classification of flowers. Continued research and development will address current challenges and unlock further opportunities.

![image](https://github.com/cavadibrahimli1/Flower_Recognition_using_CNN_MobilenetV2/assets/76445357/b1efca57-c9b5-46d3-bc6f-569b4c89b527)


## References

1. Subash. S. I, Muthiah. M. A., and N. Mathan. A novel and efficient cbir using cnn for flowers. In 2023 International Conference on Wireless Communications Signal Processing and Networking (WiSPNET), pages 1–6, 2023.
2. Hitesh Kumar Sharma, Tanupriya Choudhury, Rishi Madan, Roohi Sille, and Hussain Mahdi. Deep learning based model for multi-classification of flower images. In Vijay Singh Rathore, Vincenzo Piuri, Rosalina Babo, and Marta Campos Ferreira, editors, Emerging Trends in Expert Applications and Security, pages 393–402, Singapore, 2023. Springer Nature Singapore.
3. Mastura Hanafiah, Mohd Azraei Adnan, Shuzlina Abdul-Rahman, Sofianita Mutalib, Ariff Md Ab Malik, and Mohd Razif Shamsuddin. Flower recognition using deep convolutional neural networks. IOP Conference Series: Earth and Environmental Science, 1019(1):012021, apr 2022.
4. Büşra Rümeysa Mete and Tolga Ensari. Flower classification with deep cnn and machine learning algorithms. In 2019 3rd International Symposium on Multidisciplinary Studies and Innovative Technologies (ISMSIT), pages 1–5, 2019.
5. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks, 2019.
6. Chhaya Narvekar and Madhuri Rao. Flower classification using cnn and transfer learning in cnn- agriculture perspective. In 2020 3rd International Conference on Intelligent Sustainable Systems (ICISS), pages 660–664, 2020.
7. Gao Yifei, Qiu Chuxian, Xu Jiexiang, Miao Yixuan, and Teoh Teik Toe. Flower image classification based on improved convolutional neural network. In 2022 12th International Conference on Information Technology in Medicine and Education (ITME), pages 81–87, 2022.
8. Gandhinee Rajkomar and Sameerchand Pudaruth. A mobile app for the identification of flowers using deep learning. International Journal of Advanced Computer Science and Applications, 14(5), 2023.
9. Johanna Ärje, Dimitrios Milioris, Dat Thanh Tran, Jane Uhd Jepsen, Jenni Raitoharju, M. Gabbouj, Alexandros Iosifidis, and Toke Thomas Høye. Automatic flower detection and classification system using a light-weight convolutional neural network. 2019.
10. Charu Shree and Rupinder Kaur. A survey on flower detection techniques based on deep neural networking. International Journal of Engineering Research & Technology (IJERT), 8(05):–, 2019.
11. Salik Ram Khanal, Ranjan Sapkota, Dawood Ahmed, Uddhav Bhattarai, and Manoj Karkee. Machine vision system for early-stage apple flowers and flower clusters detection for precision thinning and pollination. IFAC-PapersOnLine, 56(2):8914–8919, 2023.
12. R Raja Subramanian, Narla Venkata Anand Sai Kumar, Nallamekala Syam Sundar, Nali Harsha Vardhan, Maram Uma Maheshwar Reddy, and M V Sanjay Kumar Reddy. Flowerbot: A deep learning aided robotic process to detect and pluck flowers. In 2022 6th International Conference on Electronics, Communication and Aerospace Technology, pages 1153–1157, 2022.
13. Alexander Mamaev. Flowers recognition, 2023.
14. Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.
15. Maria-Elena Nilsback and Andrew Zisserman. Delving into the whorl of flower segmentation. In British Machine Vision Conference, volume 1, pages 570–579, 2007.

