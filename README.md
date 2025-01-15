# Customer Segmentation for Online Retail Using K-Means Clustering

## Project Overview

This project demonstrates how K-Means clustering can be used for customer segmentation in the e-commerce industry. By segmenting customers based on their purchasing behavior, businesses can tailor marketing strategies and improve customer retention. This analysis uses transaction data from a UK-based online retail business to group customers into distinct segments.

## Table of Contents

1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Steps Involved](#steps-involved)
5. [Clustering Analysis](#clustering-analysis)
6. [Results and Insights](#results-and-insights)
7. [Usage Instructions](#usage-instructions)
8. [License](#license)

## Introduction

Customer segmentation involves dividing customers into distinct groups based on shared characteristics or behaviors. This helps businesses optimize marketing campaigns, improve customer service, and boost sales. K-Means clustering is a machine learning algorithm used to group customers into clusters with similar purchasing patterns.

This project applies K-Means clustering on a retail dataset and groups customers into clusters to understand their purchasing behaviors better.

## Technologies Used

- Python
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Dataset

The dataset used in this project contains transactional data from a UK-based online retailer. It includes information about customer purchases between December 1, 2010, and December 9, 2011.

Columns in the dataset include:

- **InvoiceNo**: Unique identifier for each transaction.
- **CustomerID**: Unique identifier for each customer.
- **StockCode**: Code for the item purchased.
- **Description**: Product description.
- **Quantity**: Number of items purchased.
- **InvoiceDate**: Date and time of the transaction.
- **UnitPrice**: Price of the item.
- **Country**: Country where the customer is located.

## Steps Involved

### 1. Data Inspection and Cleaning
- Load and inspect the dataset.
- Handle missing values and duplicates.
- Clean data by removing irrelevant columns and correcting inconsistencies.

### 2. Feature Engineering
- Calculate Recency, Frequency, and Monetary (RFM) values for each customer.
    - **Recency**: Time since the last purchase.
    - **Frequency**: Number of transactions.
    - **Monetary**: Total amount spent.

### 3. Data Standardization
- Standardize the data to ensure each feature contributes equally to the clustering process.

### 4. K-Means Clustering
- Apply the K-Means algorithm to segment customers into clusters.
- Use the Elbow Method to determine the optimal number of clusters.

### 5. Interpret the Results
- Analyze the characteristics of each customer segment.
- Visualize the clusters and identify trends in purchasing behavior.

## Clustering Analysis

### 1. Data Standardization

To ensure all features contribute equally to the distance calculation, we standardize the data:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm_data)



### 2. Applying K-Means Clustering

Using the standardized data, we apply the K-Means algorithm and use the Elbow Method to determine the optimal number of clusters:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Elbow method to find optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


### 3. Fit the K-Means Model

After determining the optimal number of clusters (let's say it's 3), we fit the K-Means model:


# Applying K-Means with 3 clusters (from elbow method)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Assign clusters to the data
rfm_data['Cluster'] = kmeans.labels_


### 4. Visualize the Clusters

To better understand the clusters, we can visualize them in a 2D plot:


import seaborn as sns

sns.scatterplot(x=rfm_data['Recency'], y=rfm_data['Monetary'], hue=rfm_data['Cluster'], palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.show()


## Results and Insights

The clustering analysis revealed three distinct customer segments based on recency, frequency, and monetary value.

- **Segment 1**: High-value, recent customers with frequent purchases.
- **Segment 2**: Low-frequency, low-monetary customers.
- **Segment 3**: High-frequency, moderate-value customers.

These insights can help the business tailor its marketing strategies and optimize product offerings to different customer segments.

## Usage Instructions

1. Clone this repository to your local machine.
2. Install the required libraries by running:
    
    pip install -r requirements.txt
    
3. Open the Jupyter Notebook (`customer_segmentation.ipynb`) and run the cells to replicate the analysis.
4. Modify the dataset and rerun the analysis for your own retail data.

## License

This project is licensed under the MIT License.


This updated version of the README file contains properly formatted code blocks for GitHub. Let me know if you need further adjustments!
