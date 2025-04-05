import argparse
from model_training import train_models
from predict import predict_churn
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Customer Churn Prediction System')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train churn prediction models')
    train_parser.add_argument('--data', help='Path to training data', default=None)

    predict_parser = subparsers.add_parser('predict', help='Make churn predictions')
    predict_parser.add_argument('--input', 
                              help='Input data (CSV file or JSON string). Omit for interactive mode')
    predict_parser.add_argument('--output', 
                              help='Output file for predictions (optional)')

    args = parser.parse_args()

    if args.command == 'train':
        print("Training churn prediction models...")
        results = train_models()
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} Results:")
            print(metrics['classification_report'])
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    elif args.command == 'predict':
        print("Making churn predictions...")
        try:
            predictions = predict_churn(args.input)
            
            if args.output:
                pd.DataFrame(predictions).to_csv(args.output, index=False)
                print(f"Predictions saved to {args.output}")
            else:
                for pred in predictions:
                    print(f"\nCustomer: {pred['customer_id']}")
                    print(f"Prediction: {pred['churn_prediction']}")
                    print(f"Probability: {pred['churn_probability']:.2f}")
                    print(f"Risk Level: {pred['churn_risk']}")
        except Exception as e:
            print(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    main()
