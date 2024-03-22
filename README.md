# Sparks-project**README**

## Predicting Student Scores using Supervised Machine Learning

This project is a part of the Data Science and Business Analytics Internship offered by The Sparks Foundation, undertaken by Soumyajyoti Chatterjee in January 2024. The task involves predicting student scores based on the number of study hours using supervised machine learning techniques.

### Project Overview:

- **Objective**: The main objective of this project is to predict the scores of students based on the number of hours they study per day.
- **Dataset**: The dataset used for this project contains two columns - 'Hours' and 'Scores'. It consists of records indicating the number of study hours and the corresponding scores achieved by students.
- **Approach**: Linear Regression, a supervised machine learning algorithm, is employed to establish a relationship between the number of study hours and the scores obtained.

### Project Workflow:

1. **Data Importing and Exploration**:
   - The necessary libraries such as Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn are imported.
   - The dataset is loaded from a CSV file hosted on GitHub.
   - Initial exploration of the dataset is performed by displaying the first few rows and visualizing the data using a scatter plot to observe the relationship between study hours and scores.

2. **Data Preparation**:
   - The dataset is divided into input features (X) and target variable (y).
   - Data splitting is carried out into training and testing sets to evaluate the performance of the model.

3. **Model Training**:
   - A Linear Regression model is instantiated and trained using the training data.

4. **Model Evaluation**:
   - The trained model is evaluated using the testing data.
   - Evaluation metrics such as R-squared (R²) and Root Mean Squared Error (RMSE) are calculated to assess the model's performance.

5. **Prediction**:
   - Using the trained model, predictions are made on the testing data.
   - Additionally, a prediction is made for a student studying 9.25 hours/day.

### Files Included:

- **predict_student_scores.ipynb**: Jupyter Notebook containing the Python code for data loading, preprocessing, model training, evaluation, and prediction.
- **README.md**: The README file providing an overview of the project, its objectives, workflow, and files included.
- **student_scores.csv**: CSV file containing the dataset used for training and testing the model.



### Dependencies:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


### Acknowledgments:

This project is completed as a part of the internship program provided by The Sparks Foundation. Special thanks to The Sparks Foundation for providing the opportunity to work on practical data science and analytics projects.
