# CMPT 353 Gait Analysis Project 

## Project Description  
This project is designed for analyzing human gait patterns using acceleration data. It allows users to upload CSV files containing sensor data (typically from accelerometers) and train machine learning models to classify different gait activities such as walking, running, and stair climbing. The application provides a simple web interface powered by Flask, and it uses various data preprocessing and feature extraction techniques to build and evaluate classification models.  

Supported models include K-Nearest Neighbors, Random Forest, and Gaussian Naive Bayes. The system applies a Butterworth filter to reduce noise and smooth the data.  

## Setup  
### Clone the Repository:  
`git clone https://github.com/rubadub13/cmpt353-project`  

### Dependencies:  
`pip install flask numpy pandas matplotlib scipy scikit-learn joblib`  

## How to use  
### Run local server:  
To run the project, open a terminal and navigate to the root directory of the project using cd. Then, start the application by running:  
`python app.py`  
Once the server starts, you will see a message like “Running on http://127.0.0.1:5000” in the terminal. Copy this URL and paste it into your browser to access the application.  

### Train and use models:  
To use the system, start by going to the **"Train"** section of the web interface, where you can upload a CSV file containing training data. The file must include the following columns: `timestamp`, `ax`, `ay`, and `az`. These acceleration values can be collected using tools like the Physics Toolbox Sensor Suite app. Once uploaded, the data will be saved to the `training_data` folder. You can then choose a training method (K-Nearest Neighbors, Random Forest, or Naive Bayes) to train a model, which will be saved to the `models` directory. After training, navigate to the **"Analyze"** section, where you can upload another CSV file to analyze and classify gait patterns using the trained model, as well as view test results and predictions.
