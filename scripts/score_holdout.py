import sys

from models.predict import predict_frame
from preprocessing.data import load_holdout


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    data = load_holdout(input_path)
    predictions = predict_frame(data)
    predictions.to_csv(output_path, index=False)
    print(len(predictions), output_path)


main()
