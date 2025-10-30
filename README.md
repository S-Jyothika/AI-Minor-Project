# AI Minor Project — Click-Through Rate (CTR) Prediction

## Project Overview
This project predicts whether a user will **click on an online advertisement** based on their online behavior and demographic data.  
The main goal is to help advertisers **improve ad targeting** and **increase Click-Through Rate (CTR)** using data-driven insights.

The project was implemented in **Python (Google Colab)** using **Logistic Regression** from Scikit-learn.

## Technologies Used
- Python  
- Google Colab  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

## Dataset Details 
The dataset used is `advertising.csv`, which contains information about user demographics and ad interaction.  
Below are the key columns:

| Feature -- Description |
|----------|-------------|
| Daily Time Spent on Site -- Time (in minutes) a user spends on the website daily |
| Age -- Age of the user |
| Area Income -- Average income of the user's area |
| Daily Internet Usage -- Average daily Internet usage time |
| Male -- Gender (1 = Male, 0 = Female) |
| Timestamp -- Date and time of record |
| Clicked on Ad -- Target variable (1 = Clicked, 0 = Not Clicked) |

## Project Workflow
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

## Model Evaluation Results
Sample results obtained from the notebook:
   - Accuracy: 95%
   - Precision (Clicked=1): 0.97
   - Recall (Clicked=1): 0.93
   - F1-Score: 0.95

## The model performed well in distinguishing between users who clicked and those who didn’t.

## Run the Project
You can open and execute the notebook directly in Google Colab using the link below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S-Jyothika/AI-Minor-Project/blob/main/CTR.ipynb)

## Future Enhancements
- Experiment with models like Decision Tree, Random Forest, and XGBoost.  
- Add feature engineering to improve prediction accuracy.  
- Create a Streamlit web app for user input and real-time CTR prediction.  
- Perform hyperparameter tuning for optimization and better model performance.

## Author
**Jyothika Sigirisetty**  
Bachelor of Technology in Computer Science and Engineering  
Lakireddy Balireddy College of Engineering, Mylavaram  

**Email:** 
   - jyothikasigirisetty@gmail.com  
**GitHub:**
   - [S-Jyothika](https://github.com/S-Jyothika)  

## Project Description
This project, titled **“Click-Through Rate (CTR) Prediction using Logistic Regression,”**  
was developed as a **self-initiated Artificial Intelligence Minor Project**.  
It focuses on predicting whether a user will click on an advertisement based on behavioral and demographic factors.  
By analyzing user data such as age, time spent online, and income level,  
the model helps advertisers improve their marketing strategies and target potential customers more effectively.

## License
This project was developed as part of the **AI Minor Project** coursework for educational purposes.

