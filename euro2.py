import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sci_stats
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import lilliefors

class CoinWeightProcessor:
    def __init__(self, data_file):
        self.coin_data = self._read_dataset(data_file)
        self._format_dataset()

    def _read_dataset(self, data_file):
        """Считывает данные из файла с учетом разных форматов разделителей"""
        try:
            return pd.read_csv(data_file, sep="\t", header=None,
                              names=["CoinID", "Mass", "Group"])
        except:
            return pd.read_csv(data_file, delim_whitespace=True,
                              names=["CoinID", "Mass", "Group"])

    def _format_dataset(self):
        """Приводит данные к нужному формату"""
        self.coin_data['Group'] = self.coin_data['Group'].astype(str)

    def process_mass_distribution(self):
        """Проводит анализ распределения массы монет"""
        print("=== Mass Distribution Analysis ===")

        # Анализ для всех монет
        self._process_subset(self.coin_data['Mass'], "All Coins")

        # Анализ по группам
        for group_id, subset in self.coin_data.groupby('Group'):
            self._process_subset(subset['Mass'], f"Group {group_id}")

        # Сравнительная визуализация
        self._visualize_groups()

    def _process_subset(self, data_subset, subset_name):
        """Обрабатывает данные для одной группы"""
        # Визуализация распределения
        self._display_distribution(data_subset, subset_name)

        # Проверка нормальности
        self._assess_normality(data_subset, subset_name)

        # Вычисление доверительных интервалов
        self._compute_confidence_intervals(data_subset, subset_name)

    def _assess_normality(self, data_subset, subset_name):
        """Проверяет нормальность распределения с помощью тестов"""
        print(f"\nNormality Tests for {subset_name}:")

        normality_tests = [
            ('Shapiro-Wilk', sci_stats.shapiro),
            ('Kolmogorov-Smirnov', lambda x: sci_stats.kstest(x, 'norm', args=(x.mean(), x.std()))),
            ('Lilliefors', lilliefors)
        ]

        for test_name, test_function in normality_tests:
            try:
                statistic, p_value = test_function(data_subset.dropna())
                print(f"{test_name}: p-value = {p_value:.4f} ({'Normal' if p_value > 0.05 else 'Not Normal'})")
            except Exception as error:
                print(f"{test_name}: Failed to execute - {str(error)}")

    def _compute_confidence_intervals(self, data_subset, subset_name):
        """Вычисляет доверительные интервалы для среднего и стандартного отклонения"""
        sample_size = len(data_subset)
        average_mass = data_subset.mean()
        std_dev = data_subset.std()

        # ДИ для среднего (t-распределение)
        t_critical = sci_stats.t.ppf(0.975, df=sample_size - 1)
        ci_mean_manual = (average_mass - t_critical * std_dev / np.sqrt(sample_size),
                         average_mass + t_critical * std_dev / np.sqrt(sample_size))

        # ДИ для среднего (statsmodels)
        ci_mean_statsmodels = sms.DescrStatsW(data_subset).tconfint_mean(alpha=0.05)

        # ДИ для стандартного отклонения
        chi2_lower = sci_stats.chi2.ppf(0.025, df=sample_size - 1)
        chi2_upper = sci_stats.chi2.ppf(0.975, df=sample_size - 1)
        ci_std_dev = (np.sqrt((sample_size - 1) * std_dev ** 2 / chi2_upper),
                      np.sqrt((sample_size - 1) * std_dev ** 2 / chi2_lower))

        print(f"\nConfidence Intervals for {subset_name}:")
        print(f"Average Mass: {average_mass:.5f} g")
        print(f"  t-distribution: [{ci_mean_manual[0]:.5f}, {ci_mean_manual[1]:.5f}]")
        print(f"  statsmodels: [{ci_mean_statsmodels[0]:.5f}, {ci_mean_statsmodels[1]:.5f}]")
        print(f"Standard Deviation: {std_dev:.5f} g, 95% CI: [{ci_std_dev[0]:.5f}, {ci_std_dev[1]:.5f}]")

    def _display_distribution(self, data_subset, title_text):
        """Отображает распределение массы с доверительными интервалами"""
        plt.figure(figsize=(12, 5))

        # Гистограмма с нормальной кривой
        plt.subplot(1, 2, 1)
        sns.histplot(data_subset, kde=True, stat='density', color='lightcoral', alpha=0.5)

        average_mass = data_subset.mean()
        std_dev = data_subset.std()
        x_range = np.linspace(data_subset.min(), data_subset.max(), 100)
        plt.plot(x_range, sci_stats.norm.pdf(x_range, average_mass, std_dev), 'r-', lw=2)

        # Доверительные интервалы
        sample_size = len(data_subset)
        t_critical = sci_stats.t.ppf(0.975, df=sample_size - 1)
        ci_mean = (average_mass - t_critical * std_dev / np.sqrt(sample_size),
                   average_mass + t_critical * std_dev / np.sqrt(sample_size))

        plt.axvline(average_mass, color='blue', linestyle='--')
        plt.axvline(ci_mean[0], color='purple', linestyle=':')
        plt.axvline(ci_mean[1], color='purple', linestyle=':')
        plt.axvspan(average_mass - std_dev, average_mass + std_dev, color='gray', alpha=0.15)

        plt.title(f'Mass Distribution ({title_text})')

        # Q-Q график
        plt.subplot(1, 2, 2)
        sci_stats.probplot(data_subset, dist="norm", plot=plt)
        plt.title(f'Quantile-Quantile Plot ({title_text})')

        plt.tight_layout()
        plt.show()

    def _visualize_groups(self):
        """Создает сравнительные графики для групп"""
        plt.figure(figsize=(14, 5))

        # 1. Плотности распределений
        plt.subplot(1, 2, 1)
        for group_id, subset in self.coin_data.groupby('Group'):
            sns.kdeplot(subset['Mass'], label=f'Group {group_id}')
        plt.title('Group Mass Distributions')
        plt.xlabel('Mass (g)')
        plt.legend()

        # 2. Средние значения с доверительными интервалами
        plt.subplot(1, 2, 2)
        group_metrics = []
        for group_id, subset in self.coin_data.groupby('Group'):
            data = subset['Mass']
            sample_size = len(data)
            average_mass = data.mean()
            std_dev = data.std()
            t_critical = sci_stats.t.ppf(0.975, df=sample_size - 1)
            ci_lower = average_mass - t_critical * std_dev / np.sqrt(sample_size)
            ci_upper = average_mass + t_critical * std_dev / np.sqrt(sample_size)
            group_metrics.append({'group': group_id, 'avg': average_mass,
                                 'ci_lower': ci_lower, 'ci_upper': ci_upper})

        metrics_df = pd.DataFrame(group_metrics)
        plt.errorbar(metrics_df['group'], metrics_df['avg'],
                     yerr=[metrics_df['avg'] - metrics_df['ci_lower'],
                           metrics_df['ci_upper'] - metrics_df['avg']],
                     fmt='o', capsize=5)
        plt.title('Average Mass with 95% Confidence Intervals')
        plt.xlabel('Group')
        plt.ylabel('Mass (g)')
        plt.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.show()

# Запуск анализа
if __name__ == "__main__":
    processor = CoinWeightProcessor("euroweight.dat.txt")
    processor.process_mass_distribution()