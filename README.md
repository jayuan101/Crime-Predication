#NYC Crime Prediction Using Machine Learning

This project leverages machine learning techniques to predict crime occurrences, aiming to assist law enforcement agencies in proactive policing and resource allocation.

📁 Project Structure

The repository contains the following key files:

Project_Final1.ipynb: The main Jupyter Notebook containing the data analysis, model training, and evaluation.

requirements.txt: A file listing the necessary Python packages to run the project.

website_app.py: A Python script for deploying the model as a web application.

⚙️ Setup Instructions

Clone the Repository

git clone https://github.com/jayuan101/Crime-Predication.git
cd Crime-Predication


Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install Required Packages

pip install -r requirements.txt


Run the Jupyter Notebook

jupyter notebook Project_Final1.ipynb


Deploy the Web Application

python website_app.py


This will start a local server, and you can access the application via http://127.0.0.1:5000/ in your web browser.

📚 Dataset

The project utilizes a crime dataset (specific details to be added), which includes information such as:

Crime type

Location (latitude, longitude)

Date and time of occurrence

Other relevant features

The dataset is preprocessed to handle missing values, encode categorical variables, and normalize numerical features before training machine learning models.
SpringerOpen

🧪 Machine Learning Models

Various machine learning algorithms are employed to predict crime occurrences, including:
ResearchGate
+3
SpringerOpen
+3
SpringerLink
+3

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)
ResearchGate
+5
SpringerOpen
+5
PhD Direction
+5

Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

🌐 Web Application

The website_app.py script uses Flask to create a web interface where users can input parameters (e.g., location, time) to predict the likelihood of crime in a given area.

📌 Notes

Ensure that all dependencies are installed as per the requirements.txt file.

The web application is designed for local deployment and may require additional configuration for production environments.

For any issues or contributions, please refer to the repository's issue tracker.
