import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sci_stats
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import lilliefors

class FlowerPetalAnalyzer:
    def __init__(self, data_file):
        self.dataset = self._import_dataset(data_file)

    def _import_dataset(self, data_file):
        return pd.read_csv(data_file, header=None, names=[
            "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"
        ])

    def evaluate_petal_length(self):
        print("=== Petal Length Evaluation ===")

        for species, subset in self.dataset.groupby("Species"):
            self._process_subset(subset["PetalLength"], f"Species {species}")

        self._compare_distributions("PetalLength", "Petal Length")

    def _process_subset(self, data_subset, subset_label):
        self._visualize_distribution(data_subset, subset_label)

        self._check_normality(data_subset, subset_label)

        self._estimate_intervals(data_subset, subset_label)

    def _check_normality(self, data_subset, subset_label):
        print(f"\nNormality Assessment for {subset_label}:")

        normality_checks = [
            ('Shapiro-Wilk', sci_stats.shapiro),
            ('Kolmogorov-Smirnov', lambda x: sci_stats.kstest(x, 'norm', args=(x.mean(), x.std()))),
            ('Lilliefors', lilliefors)
        ]

        for check_name, check_function in normality_checks:
            try:
                statistic, p_value = check_function(data_subset.dropna())
                result = "Normal" if p_value > 0.05 else "Not Normal"
                print(f"{check_name}: p-value = {p_value:.4f} ({result})")
            except Exception as error:
                print(f"{check_name}: Failed to execute - {str(error)}")

    def _estimate_intervals(self, data_subset, subset_label):
        sample_size = len(data_subset)
        average = data_subset.mean()
        std_dev = data_subset.std()

        ci_mean = sms.DescrStatsW(data_subset).tconfint_mean(alpha=0.05)

        chi2_lower = sci_stats.chi2.ppf(0.025, df=sample_size - 1)
        chi2_upper = sci_stats.chi2.ppf(0.975, df=sample_size - 1)
        ci_std_dev = (
            np.sqrt((sample_size - 1) * std_dev ** 2 / chi2_upper),
            np.sqrt((sample_size - 1) * std_dev ** 2 / chi2_lower)
        )

        print(f"\nConfidence Intervals for {subset_label}:")
        print(f"Average: {average:.2f} cm, 95% CI: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
        print(f"Standard Deviation: {std_dev:.2f} cm, 95% CI: [{ci_std_dev[0]:.2f}, {ci_std_dev[1]:.2f}]")

    def _visualize_distribution(self, data_subset, title_text):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        sns.histplot(data_subset, kde=True, stat='density', color='lightcoral', alpha=0.5)
        average = data_subset.mean()
        std_dev = data_subset.std()
        x_range = np.linspace(data_subset.min(), data_subset.max(), 100)
        plt.plot(x_range, sci_stats.norm.pdf(x_range, average, std_dev), 'r-', lw=2)
        plt.title(f'Distribution ({title_text})')

        plt.subplot(1, 2, 2)
        sci_stats.probplot(data_subset, dist="norm", plot=plt)
        plt.title(f'Quantile Plot ({title_text})')

        plt.tight_layout()
        plt.show()

    def _compare_distributions(self, feature_name, display_name):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for species, subset in self.dataset.groupby("Species"):
            sns.kdeplot(subset[feature_name], label=f'{species}')
        plt.title(f'{display_name} Comparison Across Species')
        plt.xlabel('Length (cm)')
        plt.legend()

        plt.subplot(1, 2, 2)
        group_metrics = []
        for species, subset in self.dataset.groupby("Species"):
            data = subset[feature_name]
            sample_size = len(data)
            average = data.mean()
            std_dev = data.std()
            t_critical = sci_stats.t.ppf(0.975, df=sample_size - 1)
            ci_lower = average - t_critical * std_dev / np.sqrt(sample_size)
            ci_upper = average + t_critical * std_dev / np.sqrt(sample_size)
            group_metrics.append({
                'species': species,
                'average': average,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

        metrics_df = pd.DataFrame(group_metrics)
        plt.errorbar(metrics_df['species'], metrics_df['average'],
                     yerr=[metrics_df['average'] - metrics_df['ci_lower'],
                           metrics_df['ci_upper'] - metrics_df['average']],
                     fmt='o', capsize=5, color='blue', ecolor='gray')
        plt.title(f'Average {display_name} with 95% CI')
        plt.xlabel('Species')
        plt.ylabel('Length (cm)')
        plt.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    processor = FlowerPetalAnalyzer("iris.txt")
    processor.evaluate_petal_length()
