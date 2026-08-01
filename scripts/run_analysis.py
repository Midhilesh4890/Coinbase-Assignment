from preprocessing.data import load_dataset
from statistical_analysis.plots import plot_class_distribution, plot_template_frequency
from statistical_analysis.statistics import class_distribution, duplication_stats, label_conflicts


def main():
    data = load_dataset()
    print(class_distribution(data))
    print(duplication_stats(data))
    print(label_conflicts(data))
    plot_class_distribution(data, "reports/class_distribution.png")
    plot_template_frequency(data, "reports/template_duplication.png")
    print("reports/class_distribution.png")
    print("reports/template_duplication.png")


main()
