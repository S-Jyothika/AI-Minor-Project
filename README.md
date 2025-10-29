# 🧠 AI Minor Project — Click-Through Rate (CTR) Prediction

## 📌 Project Overview
This project predicts whether a user will **click on an online advertisement** based on their online behavior and demographic data.  
The main goal is to help advertisers **improve ad targeting** and **increase Click-Through Rate (CTR)** using data-driven insights.

The project was implemented in **Python (Google Colab)** using **Logistic Regression** from Scikit-learn.

---

## 🧮 Technologies Used
- Python  
- Google Colab  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## 📊 Dataset Details
The dataset used is `advertising.csv`, which contains information about user demographics and ad interaction.  
Below are the key columns:

| Feature | Description |
|----------|-------------|
| Daily Time Spent on Site | Time (in minutes) a user spends on the website daily |
| Age | Age of the user |
| Area Income | Average income of the user's area |
| Daily Internet Usage | Average daily Internet usage time |
| Male | Gender (1 = Male, 0 = Female) |
| Timestamp | Date and time of record |
| Clicked on Ad | Target variable (1 = Clicked, 0 = Not Clicked) |

---

## ⚙️ Project Workflow
1. **Data Import and Loading**  
   - Loaded the dataset using Pandas and explored the structure.
2. **Data Exploration and Cleaning**  
   - Checked for missing values and understood feature relationships.
3. **EDA (Exploratory Data Analysis)**  
   - Used Seaborn and Matplotlib to visualize Age, Internet Usage, and Click patterns.
4. **Data Preprocessing**  
   - Selected important features and applied data scaling.
5. **Model Training (Logistic Regression)**  
   - Built a Logistic Regression model to classify whether a user clicks on an ad.
6. **Model Evaluation**  
   - Measured model performance using:
     - Accuracy
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-score)

---

## 📈 Model Evaluation Results
Sample results obtained from the notebook:

