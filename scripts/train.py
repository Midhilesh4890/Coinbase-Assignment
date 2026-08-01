from preprocessing.data import load_dataset
from models.evaluate import cross_validate, log_experiment


def main():
    data = load_dataset()
    result = cross_validate(data, "tfidf_logreg")
    print(result["macro_f1_mean"])
    print(result["macro_f1_std"])
    print(result["fraud_recall_mean"])
    print(result["fraud_recall_std"])
    print(result["per_fold"])
    print(result["report"])
    log_experiment(result)


main()
