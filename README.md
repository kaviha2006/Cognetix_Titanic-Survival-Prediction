🌊 Titanic Survival Prediction – Machine Learning Project

This project is created as part of the Cognetix Technology Internship – Foundational Stage. The goal of this project is to predict whether a passenger survived the Titanic disaster using demographic and travel-related information.

📌 Project Overview

The Titanic dataset contains historical data of passengers aboard the Titanic, including survival status and personal details such as age, gender, and travel class.

Key features used in this project include:

Passenger Class (Pclass)

Sex

Age

Number of Siblings/Spouses aboard (SibSp)

Number of Parents/Children aboard (Parch)

Fare

Embarkation Port

Using these features, machine learning classification models are trained to predict passenger survival. The model accepts user input through the terminal and displays whether the passenger is predicted to have survived.

📂 Dataset

The dataset used in this project:

🔗 https://www.kaggle.com/c/titanic/data

File used: train.csv

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

🚀 How to Run the Project

Install required libraries:

pip install pandas numpy scikit-learn matplotlib

Run the script:

python titanic_prediction.py

====== TITANIC SURVIVAL PREDICTION ======

Enter passenger details to predict survival status.

Enter Pclass (1/2/3): 3

Enter Sex (male/female): female

Enter Age: 22

Enter number of siblings/spouses aboard: 1

Enter number of parents/children aboard: 0

Enter Fare: 7.25

Enter Embarked (C/Q/S): S

✅ Output:

Prediction: Survived

📊 Model Performance

Two classification models were implemented:

Logistic Regression

Random Forest Classifier

The Random Forest model achieved higher accuracy and better overall performance, and is used for final predictions. Model evaluation was performed using accuracy, precision, recall, and F1-score metrics.

📈 Data Visualization

The project includes visual analysis of survival rates based on:

Gender

Passenger Class

Age Group

🎯 Internship Task

This project is submitted as Task 3 – Titanic Survival Prediction under the Foundational Stage of the Cognetix Technology Internship Program.
