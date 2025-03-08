# Credit-Card-Fraud-Detection
## Fraud Detection System using Autoencoder Neural Network

## Overview
The Fraud Detection System using Autoencoder Neural Network is a machine learning project aimed at identifying suspicious credit card transactions and preventing fraudulent activities. The project leverages anomaly detection techniques and handles class imbalance in the data to achieve accurate fraud detection.

## Purpose
The primary purpose of this project is to develop a robust and efficient fraud detection system that can automatically detect and flag fraudulent credit card transactions. By using an unsupervised or semi-supervised approach, the system can identify anomalies in the data without requiring explicit fraud labels during training.

## Key Features
- Utilizes an Autoencoder Neural Network to learn and reconstruct normal transaction patterns.
- Detects anomalies in credit card transactions based on the deviation from normal patterns.
- Handles class imbalance in the dataset to improve the performance of the fraud detection model.
- Uses precision-recall curves to evaluate the model's performance in handling true positives and false positives.
- Provides insights into the reconstruction error distribution for different transaction classes.
- Applies a threshold-based approach to classify new transactions as normal or fraudulent.

## Setup and Dependencies
To run this project, you will need the following libraries and dependencies:
- TensorFlow 1.2
- Keras 2.0.4
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Ensure that the required libraries are installed before running the code.

## Dataset
The credit card transaction dataset used in this project is available for download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains information on credit card transactions conducted over a two-day period, with 492 fraudulent transactions out of a total of 284,807 transactions. The dataset has been anonymized and transformed using PCA for privacy reasons.

The dataset contains the following columns:
- Time: The seconds elapsed between each transaction and the first transaction in the dataset.
- Amount: The transaction amount.
- V1, V2, ..., V28: The transformed features obtained using PCA for privacy reasons.
- Class: The target variable, where 0 represents a normal transaction and 1 represents a fraudulent transaction.

## Exploration and Preprocessing
The project starts with an exploratory data analysis to understand the data's characteristics and distribution. It checks for missing values, visualizes the class distribution, examines the amount used in different transaction classes, and explores the time of transactions. After exploration, the data is preprocessed by dropping the Time column and scaling the Amount column using StandardScaler.

## Autoencoders for Anomaly Detection
Autoencoders are employed as the core of the fraud detection system. The model is trained to approximate the identity function, reconstructing the input data from the same input. The Autoencoder Neural Network learns a compressed representation of the data, which helps in identifying anomalies or deviations from normal patterns.

## Model Training and Evaluation
The model is trained using only normal transactions, creating a one-class classification scenario for anomaly detection. The training progress is monitored using TensorBoard. The performance of the trained model is evaluated using reconstruction error and precision-recall curves.

## Performance Matrices
The following performance metrics are used to evaluate the fraud detection system:
- Precision: The ratio of true positives to the sum of true positives and false positives. High precision indicates fewer false positives.
- Recall: The ratio of true positives to the sum of true positives and false negatives. High recall indicates fewer false negatives.
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure of model performance.
- Confusion Matrix: A table that shows the true positive, false positive, true negative, and false negative counts, providing a comprehensive view of the model's performance.
- ROC Curve: The receiver operating characteristic curve, plotting true positive rate against false positive rate, to visualize the trade-off between sensitivity and specificity.

## Conclusion
The Fraud Detection System using Autoencoder Neural Network demonstrates the effectiveness of using unsupervised or semi-supervised approaches for anomaly detection in credit card transactions. By leveraging Autoencoders, the system can learn and identify normal transaction patterns, allowing it to detect potential frauds efficiently.

The performance matrices show that the model achieves high recall and precision, indicating a well-balanced fraud detection system. The project provides insights into handling class imbalance and selecting appropriate thresholds for fraud classification. With a threshold-based approach, the system can accurately classify new transactions as normal or fraudulent, thereby mitigating potential losses due to fraudulent activities.

## Best Performing Model
Among the machine learning models used in this project, the Autoencoder Neural Network stands out as the best performing model. It demonstrates robust anomaly detection capabilities, effectively capturing patterns in normal transactions and flagging potential fraudulent activities with high accuracy. The model's ability to handle class imbalance and generate precise predictions contributes to its superiority in fraud detection compared to traditional supervised models.

## References
- Building Autoencoders in Keras: [link](https://blog.keras.io/building-autoencoders-in-keras.html)
- Stanford tutorial on Autoencoders: [link](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
- Stacked Autoencoders in TensorFlow: [link](https://cmgreen.io/2016/01/04/tensorflow_deep_autoencoder.html)

## License
This project is open-source and available under the [MIT License](LICENSE).
  
## Contact
Built with 💛 by [Nishant Raj](https://www.linkedin.com/in/the-nishant-raj-82972b208/).  
Connect with me:  
- **YouTube:** [the_nishant_raj](https://www.youtube.com/@the_nishant_raj)  
- **Instagram:** [the_nishant_raj](https://www.instagram.com/the_nishant_raj/)
