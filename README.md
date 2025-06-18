# Machine Learning with Python

This repository contains practical applications of various machine learning algorithms using the Python programming language and popular libraries such as `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, and `statsmodels`. The repository comprises Jupyter Notebooks focusing on supervised learning (regression and classification), unsupervised learning (clustering), natural language processing (NLP), and the fundamentals of reinforcement learning.

Each notebook provides comprehensive examples, covering data loading and preprocessing steps, model implementation, and result evaluation. Visualizations are actively used to enhance understanding of model behavior and performance.

## Covered Topics and Applications

This repository covers the following core machine learning topics and practical applications:

### 1. Regression Analysis
Various regression algorithms are explored for modeling linear and non-linear relationships.
* **Simple Linear Regression (`Simple_Linear.ipynb`):** A basic linear regression application on monthly sales data, with visualization of predictions. Data scaling and inverse transformation steps are highlighted.
* **Multiple Linear Regression (`Multiple_Linear_Reg.ipynb`, `Regression.ipynb`):** Application of multiple regression on the `tenis.csv` dataset. Categorical data handling (Label/One-Hot Encoding) and feature selection using the **Backward Elimination** method with the `statsmodels` library are demonstrated in detail. Emphasis is placed on the statistical significance of the model.
* **Polynomial Regression (`Polynominal_Regression.ipynb`):** Application and visualization of 2nd and 4th-degree polynomial regression models on the `maaslar.csv` (salary) dataset to capture non-linear relationships.
* **Support Vector Regression (SVR) (`SVR.ipynb`):** Application of the SVR model on the `maaslar.csv` (salary) dataset. The sensitivity of SVR to feature scaling and how to address it is shown.
* **Decision Tree Regression (`Decision_Tree.ipynb`):** Application of the Decision Tree Regression model on the `maaslar.csv` (salary) dataset.
* **Random Forest Regression (`Rondom_Forest.ipynb`):** Application of the Random Forest Regression model on the `maaslar.csv` (salary) dataset.
* **Regression Model Comparison and R2 Calculation (`R2_Calculation.ipynb`):** Comparing the performance of all the above regression models on the `maaslar.csv` (salary) dataset using the $R^2$ score.

### 2. Classification Applications
Fundamental classification algorithms used to categorize data:
* **Various Classification Algorithms and ROC/AUC Analysis (`Classification Applications.ipynb`):** Implementation of popular classification algorithms such as Naive Bayes, K-Nearest Neighbors, Support Vector Machine (SVM), Decision Tree, and Random Forest on the Iris dataset. A confusion matrix is generated for each model, and **ROC curves with AUC (Area Under the Curve)** values are calculated and visualized.
* **Missing Data Handling and Logistic Regression (`Missing_data.ipynb`):** Demonstrating essential data preprocessing steps (filling missing values with the mean and converting categorical data to numerical format using Label/One-Hot Encoding) on a dataset with missing values (`eksikveriler.csv`). A simple Logistic Regression model is applied to the preprocessed data.

### 3. Clustering Analysis
Discovering natural groupings within a dataset using unsupervised learning:
* **K-Means Clustering (`K_Means.ipynb`):** Customer segmentation using the K-Means algorithm on the `musteriler.csv` (customer) dataset. Determining the optimal number of clusters using the **"Elbow Method"** and visualizing the clustering results.
* **Hierarchical Clustering (`Hierarchical_Segmentation.ipynb`):** Application of Hierarchical Clustering (Agglomerative Clustering) on the same customer dataset. Visualization of clustering results and plotting a **dendrogram** to show the cluster merging process.

### 4. Natural Language Processing (NLP)
Basic NLP techniques for working with text data:
* **Sentiment Analysis Application (`NLP_Application.ipynb`, `NLP_Application_2.ipynb`):** Basic NLP preprocessing steps (lowercasing, punctuation removal, stemming, stop-word removal) applied to the `Restaurant_Reviews.tsv` dataset for sentiment analysis. A "Bag of Words" model is created from the cleaned text, a Naive Bayes classification model is trained, and performance is evaluated with a confusion matrix.

### 5. Reinforcement Learning
An application for optimizing decision-making processes:
* **Multi-Armed Bandit Problem (UCB) (`UCB.ipynb`):** Application of **Random Selection** and **Upper Confidence Bound (UCB)** algorithms to the Multi-Armed Bandit problem using ad click data (`Ads_CTR_Optimisation.csv`). It demonstrates UCB's ability to optimize for higher rewards (clicks) compared to random selection.

## Datasets
The datasets used in this project (mostly in `csv` or `tsv` format) should be located in the same directory as their respective Jupyter Notebooks:
* `Iris.csv`
* `maaslar.csv` (Salary data)
* `musteriler.csv` (Customer data)
* `eksikveriler.csv` (Missing data)
* `tenis.csv` (Tennis data)
* `Restaurant_Reviews.tsv`
* `satislar.csv` (Sales data)
* `Ads_CTR_Optimisation.csv`

## Setup and Usage
To run these notebooks, you will need the following Python libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `scipy`
* `nltk`
* `statsmodels`

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib scikit-learn scipy nltk statsmodels
import nltk
nltk.download('stopwords')
# Other necessary modules (e.g., 'punkt') can be downloaded similarly.```
```
After installation, you can launch Jupyter Notebook (`jupyter notebook`) and open and run the `.ipynb` files.

## Contribution
Feel free to contribute to this project. You can submit pull requests for bug fixes, new features, or improvements.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
