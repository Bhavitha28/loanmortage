from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
#from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LabelEncoderScratch:
    def __init__(self):
        self.classes_ = {}

    def fit(self, X):
        for i, col in enumerate(X.T):
            self.classes_[i] = {label: idx for idx, label in enumerate(np.unique(col))}
        return self

    def transform(self, X):
        X_encoded = np.zeros(X.shape, dtype=int)
        for i, col in enumerate(X.T):
            X_encoded[:, i] = [self.classes_[i][label] for label in col]
        return X_encoded

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, num_iterations=5000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = y

        for i in range(self.num_iterations):
            self.update_weights()

    def update_weights(self):
        A = self.sigmoid(np.dot(self.X, self.W) + self.b)
        tmp = (A - self.y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        Z = self.sigmoid(np.dot(X, self.W) + self.b)
        y_pred = np.where(Z > 0.5, 1, 0)
        return y_pred

scaler_path = 'scaler.pkl'
encoder_path = 'label_encoder_scratch.pkl'
model_path = 'best_logistic_regression_model.pkl'


with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(encoder_path, 'rb') as file:
    label_encoder_scratch = pickle.load(file)


with open(model_path, 'rb') as file:
    model = pickle.load(file)


numerical_columns = ['Loan_Amount_Requested', 'Annual_Income', 'Debt_To_Income',
                     'Inquiries_Last_6Mo', 'Number_Open_Accounts', 'Total_Accounts',
                     'loan_income_ratio', 'total_income_ratio', 'debt_income_ratio']
categorical_columns = ['Length_Employed', 'Income_Verified', 'Purpose_Of_Loan']


df = pd.read_csv('processed_test_data.csv')


dropdown_options = {}
for column in numerical_columns + categorical_columns:
    if column in df.columns:
        unique_values = df[column].dropna().unique().tolist()
        dropdown_options[column] = unique_values[:1000]  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  
        
        
        test_df = pd.DataFrame([data])

        
        for column in numerical_columns + categorical_columns:
            if column not in test_df.columns:
                return jsonify({'error': f'Missing column: {column}'}), 400
        
        
        for col in numerical_columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

        
        test_df[categorical_columns] = test_df[categorical_columns].astype(str)
        
        
        test_df = test_df.dropna(subset=numerical_columns + categorical_columns)
        
        X_test_encoded = label_encoder_scratch.transform(test_df[categorical_columns].values)
        
        X_test_scaled = scaler.transform(test_df[numerical_columns].values)
        
        X_test_processed = np.hstack([X_test_scaled, X_test_encoded])
        
        X_test_final = X_test_processed
        
        
        predictions = model.predict(X_test_final)
        print("Predictions:", predictions)  
        
        
        predictions = predictions.tolist()
        
        
        return jsonify({'Predicted_Interest_Rate': predictions[0]})
    
    except Exception as e:
        print("Error during prediction:", e)  
        return jsonify({'error': str(e)}), 500

@app.route('/dropdown-options', methods=['GET'])
def get_dropdown_options():
    return jsonify(dropdown_options)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
