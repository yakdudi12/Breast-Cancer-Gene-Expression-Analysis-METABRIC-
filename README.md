# Breast-Cancer-Gene-Expression-Analysis-METABRIC-
This project conducts an exploratory and statistical data analysis (EDA) of the METABRIC dataset, containing genomic and clinical data from 1,980 breast cancer samples. It examines the relationship between treatments, and patient survival, using visualizations and data processing tools to understand factors affecting breast cancer progression
In addition to analyzing treatment-survival relationships, three neural network models were implemented to predict different biological targets using gene expression (RNA-seq) data.

---

## 📊 Context  

Breast cancer is the most common form of cancer among women, with an annual incidence of over 2.1 million cases. It accounts for the highest number of cancer-related deaths in women.  

The main goal of this analysis is to **accurately estimate prognosis and survival duration**, exploring genetic differences , treatment responses and **develop predictors** to help personalize treatment, avoiding unnecessary procedures.

The METABRIC dataset, a collaboration between researchers in Canada and the United Kingdom, has been featured in leading journals such as *Nature Communications* (Pereira et al., 2016).  

---

## 🛠️ Features  

The code includes an `EDA` class that provides a variety of methods for performing exploratory data analysis. And Deep Learning implementations for predictors. Key functionalities include:  

### **Data Processing**  
- **Data Loading:** Compatible with CSV, Excel, and SQLite database formats.  
- **Missing Values Identification:** Visualizes null values with heatmaps.  
- **Duplicate Detection and Removal:** Automatically cleans duplicate data.  

### **Exploratory Analysis**  
- **Statistical Summary:** Generates descriptive statistics for numerical and categorical variables.  
- **Distribution Visualizations:** Creates boxplots and categorical distribution charts.  
- **Correlation Matrix:** Displays correlations between numerical variables.  

### **Specific Analyses**  
- **Age Analysis:** Examines the relationship between age at diagnosis and overall survival.  
- **Treatment Impact:** Visual comparisons of chemotherapy, hormone therapy, and radiotherapy in terms of survival.  
- **Mortality Analysis:** Percentual comparison of deaths across different treatments.
- **Hormonal Receptor Analysis:** Percentual comparison of deaths and tumor size across different hormonal Receptors.
- **Tumor Stage distribution:** Percentual distribution of tumor stage among the data population

### **Cancer Type Predictor**
- **Input:** Gene expression profiles (RNA-seq)
- **Output:** Detailed histological breast cancer type (cancer_type_detailed)

### **Mutated Gene Predictor**
- **Input:** Gene expression profiles (RNA-seq)
- **Output:** Prediction of relevant gene mutations

### **Pam50 + claudin low subtype Gene Predictor**
- **Input:** Gene expression profiles (RNA-seq)
- **Output:** Classification into PAM50 subtypes (LumA, LumB, Basal, Her2, etc.).
  
Each model includes:
-Preprocessing and standardization of RNA-seq data.
-Class balancing (if needed).
-MLP architecture with dense layers, batch normalization and dropout.
-Training and validation using metrics like accuracy, recall, and F1-score.
-Visualization of loss and accuracy over epochs

---

## 📁 Project Structure  

```
📂 BreastCancerEDA  
 ├── 📄 main.py                   # Main project code  
 ├── 📄 README.md                 # This file  
 ├── 📂 Breast_cancer_Plots       # Plots  
 ├── 📂 breast_cancer_dataset.zip # Zip Folder containing the METABRIC dataset
 ├── 📂 Models_plots_results      # Plots for the model results 
 └── 📂 Models                    # Folder with models scripts
```

---


## 🧰 Libraries Used  

- **Pandas:** Data manipulation.  
- **Matplotlib:** Data visualization.  
- **Seaborn:** Advanced and aesthetically pleasing visualizations.  
- **SQLite3:** Database connection and querying.  
- **NumPy:** Mathematical and statistical operations.
- **Scikit-learn:** Train/test splitting, metrics, preprocessing.
- **PyTorch:** Neural network implementation.

---

## 📈 Results  

- **Identification of Key Patterns:** Relationships between clinical features and survival outcomes.  
- **Meaningful Visualizations:** Clear and detailed charts summarizing the impact of factors such as age and treatments on survival.  
- **Data Cleaning:** A dataset ready for further analysis, including predictive modeling.
-Identification of important patterns in clinical and genomic data.
-Robust classification of cancer subtypes and gene mutations.
-Clear visualizations of model performance.

---

## ✒️ References  

1. Pereira, B., Chin, S. F., et al. (2016). "The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes." *Nature Communications*.  
2. cBioPortal: [https://www.cbioportal.org/](https://www.cbioportal.org/).  

---

## 🤝 Contributions  

This project is part of my professional portfolio. If you have suggestions or feedback, I would be happy to hear from you!  

---

## 📬 Contact  

Britez Santiago    
[LinkedIn](https://www.linkedin.com/in/santiago-luis-britez-101a8a217)  
