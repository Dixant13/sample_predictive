# Credit Card Fraud Detection: Sampling and Model Evaluation

## Project Overview
This project addresses the challenge of **imbalanced datasets** in machine learning. We use a credit card transaction dataset where the number of fraudulent transactions is significantly lower than genuine ones. The goal is to balance the data and evaluate how different sampling techniques impact the accuracy of various machine learning models.

## Workflow
1.  **Balancing**: Applied **Random Oversampling** to equate the minority class (Fraud) with the majority class (Genuine).
2.  **Sampling**: Generated five distinct samples using statistical sampling techniques.
3.  **Evaluation**: Trained and tested five different machine learning models on each sample.

---

## 1. Balancing Technique
The original dataset was highly skewed:
* **Class 0 (Genuine):** 763
* **Class 1 (Fraud):** 9

We used **Random Oversampling** to bring both classes to **763** instances each.



---

## 2. Sampling Techniques
The following five sampling methods were used to create subsets of the balanced data:

| Sample | Technique | Description |
| :--- | :--- | :--- |
| **Sample 1** | **Simple Random Sampling** | Randomly selected rows based on a 95% confidence level. |
| **Sample 2** | **Stratified Sampling** | Ensures the 50/50 class ratio is perfectly preserved in the sample. |
| **Sample 3** | **Systematic Sampling** | Selects every $k$-th element from the dataset. |
| **Sample 4** | **Cluster Sampling** | Groups data into clusters and randomly selects specific clusters. |
| **Sample 5** | **Bootstrap Sampling** | Random sampling with replacement (using a unique random seed). |



---

## 3. Machine Learning Models
Five different classifiers were used for cross-evaluation:
* **M1: Logistic Regression**
* **M2: Decision Tree**
* **M3: Random Forest**
* **M4: Support Vector Machine (SVM)**
* **M5: K-Nearest Neighbors (KNN)**

---

## 4. Results (Accuracy)
The table below shows the accuracy results of each model across the different samples:

| Model | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9610 | 0.9610 | 0.9216 | 0.9221 | 0.9000 |
| **Decision Tree** | 0.9481 | 1.0000 | 1.0000 | 0.9870 | 0.9875 |
| **Random Forest** | 0.9870 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **SVM** | 0.7532 | 0.6494 | 0.6536 | 0.7013 | 0.6500 |
| **KNN** | 0.9740 | 0.9221 | 0.9346 | 0.8831 | 0.9500 |

### Conclusions
* **Best Performing Model**: **Random Forest** showed the highest stability and accuracy across all sampling types.
* **Sampling Impact**: Stratified and Cluster sampling yielded the most consistent results for tree-based models.

