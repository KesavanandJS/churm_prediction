# Customer Churn Prediction System

## Description
This project implements a customer churn prediction system using machine learning techniques. It aims to predict whether a customer will churn based on various features such as tenure, monthly charges, and service usage.

## File Structure
- `requirements.txt`: Lists the dependencies required to run the project.
- `config.py`: Contains configuration settings, including data paths and feature definitions.
- `data_preprocessing.py`: Handles data loading, preprocessing, and feature engineering.
- `model_training.py`: Trains machine learning models and evaluates their performance.
- `predict.py`: Contains functions for making predictions on new customer data.
- `main.py`: The entry point for the application, handling command-line arguments for training and prediction.
- `data/raw/churn_data.csv`: Sample data used for training and predictions.
- `models/`: Directory where trained models are saved.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions
- To train the churn prediction models:
   ```bash
   python main.py train
   ```
- To make predictions using sample data:
   ```bash
   python main.py predict --input "data/raw/churn_data.csv"
   ```
- To use interactive input for predictions:
   ```bash
   python main.py predict
   ```

## Expected Output
The output will display the churn prediction for each customer, including:
- Customer ID
- Churn Prediction (Yes/No)
- Churn Probability
- Churn Risk Level (High/Medium/Low)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
