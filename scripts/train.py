from preprocessing.data import load_dataset
from models.evaluate import confusion, cross_validate, log_experiment


def main():
    data = load_dataset()
    for name in ["tfidf_logreg", "tfidf_logreg_balanced"]:
        result = cross_validate(data, name)
        print(name)
        print(result["macro_f1_mean"])
        print(result["macro_f1_std"])
        print(result["fraud_recall_mean"])
        print(result["fraud_recall_std"])
        print(result["per_fold"])
        print(result["report"])
        print(confusion(data, name))
        log_experiment(result)


main()
