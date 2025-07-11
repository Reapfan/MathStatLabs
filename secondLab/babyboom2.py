import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import lilliefors

def import_dataset(file_path):
    column_ranges = [(1, 8), (9, 16), (17, 24), (25, 32)]
    column_names = ['Timestamp', 'Gender', 'WeightAtBirth', 'MinutesSinceMidnight']

    records_list = []
    with open(file_path, 'r') as source_file:
        for line in source_file:
            row_data = [line[start - 1:end].strip() for start, end in column_ranges]
            records_list.append(row_data)

    records = pd.DataFrame(records_list, columns=column_names)

    records['WeightAtBirth'] = pd.to_numeric(records['WeightAtBirth'].str.replace(' ', ''), errors='coerce')
    records['MinutesSinceMidnight'] = pd.to_numeric(records['MinutesSinceMidnight'].str.replace(' ', ''), errors='coerce')
    records['Gender'] = records['Gender'].astype(int)

    return records

def analyze_distribution(data_series, group_name, significance_level=0.05):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(data_series, kde=True, color='lightcoral', stat='density')
    x_values = np.linspace(data_series.min(), data_series.max(), 100)
    plt.plot(x_values, stats.norm.pdf(x_values, data_series.mean(), data_series.std()), 'r-', lw=2)
    plt.title(f'Distribution Plot ({group_name})')

    plt.subplot(1, 2, 2)
    stats.probplot(data_series, dist="norm", plot=plt)
    plt.title(f'Quantile Plot ({group_name})')

    plt.tight_layout()
    plt.show()

    shapiro_stat, shapiro_pval = stats.shapiro(data_series)
    ks_stat, ks_pval = stats.kstest(data_series, 'norm', args=(data_series.mean(), data_series.std()))
    lillie_stat, lillie_pval = lilliefors(data_series)

    print(f"\nDistribution Analysis for {group_name}:")
    print(f"- Shapiro-Wilk Test: p-value = {shapiro_pval:.4f} → {'Normal' if shapiro_pval > significance_level else 'Not Normal'}")
    print(f"- Kolmogorov-Smirnov Test: p-value = {ks_pval:.4f} → {'Normal' if ks_pval > significance_level else 'Not Normal'}")
    print(f"- Lilliefors Test: p-value = {lillie_pval:.4f} → {'Normal' if lillie_pval > significance_level else 'Not Normal'}")

    confidence_interval_mean = sms.DescrStatsW(data_series).tconfint_mean(alpha=significance_level)
    sample_size = len(data_series)
    chi2_upper = stats.chi2.ppf(1 - significance_level / 2, sample_size - 1)
    chi2_lower = stats.chi2.ppf(significance_level / 2, sample_size - 1)
    confidence_interval_std = (
        np.sqrt((sample_size - 1) * data_series.std() ** 2 / chi2_upper),
        np.sqrt((sample_size - 1) * data_series.std() ** 2 / chi2_lower)
    )

    print(f"\n95% Confidence Intervals for {group_name}:")
    print(f"- Mean: ({confidence_interval_mean[0]:.2f}, {confidence_interval_mean[1]:.2f})")
    print(f"- Standard Deviation: ({confidence_interval_std[0]:.2f}, {confidence_interval_std[1]:.2f})")

def verify_exponential(intervals, significance_level=0.05):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(intervals, kde=True, stat='density', color='lightcoral')
    x_values = np.linspace(0, intervals.max(), 100)
    plt.plot(x_values, stats.expon.pdf(x_values, scale=intervals.mean()), 'r-', lw=2)
    plt.title('Time Interval Distribution')

    plt.subplot(1, 2, 2)
    stats.probplot(intervals, dist="expon", sparams=(intervals.mean(),), plot=plt)
    plt.title('Exponential Quantile Plot')
    plt.tight_layout()
    plt.show()

    rate_estimate = 1 / intervals.mean()
    ks_stat, ks_pval = stats.kstest(intervals, 'expon', args=(0, 1 / rate_estimate))

    print(f"\nExponential Distribution Test:")
    print(f"- Estimated Rate: {rate_estimate:.4f} events per minute")
    print(f"- KS Test: p-value = {ks_pval:.4f} → {'Exponential' if ks_pval > significance_level else 'Not Exponential'}")

def test_poisson_distribution(events_per_hour, significance_level=0.05):
    """Проверяет, соответствует ли число событий в час распределению Пуассона"""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(events_per_hour, discrete=True, stat='probability', color='lightcoral')
    plt.title('Hourly Events Distribution')

    poisson_rate = events_per_hour.mean()
    observed_frequencies = events_per_hour.value_counts().sort_index()

    max_value = observed_frequencies.index.max()
    all_frequencies = pd.Series(0, index=range(max_value + 1))
    all_frequencies.update(observed_frequencies)

    total_observed = all_frequencies.sum()
    expected_frequencies = [stats.poisson.pmf(k, poisson_rate) * total_observed for k in all_frequencies.index]

    grouped_observed = []
    grouped_expected = []
    temp_observed = temp_expected = 0

    for obs, exp in zip(all_frequencies, expected_frequencies):
        if exp < 5:
            temp_observed += obs
            temp_expected += exp
        else:
            if temp_observed > 0:
                grouped_observed.append(temp_observed)
                grouped_expected.append(temp_expected)
                temp_observed = temp_expected = 0
            grouped_observed.append(obs)
            grouped_expected.append(exp)

    if temp_observed > 0:
        grouped_observed.append(temp_observed)
        grouped_expected.append(temp_expected)

    if not np.isclose(sum(grouped_observed), sum(grouped_expected)):
        grouped_expected = [exp * sum(grouped_observed) / sum(grouped_expected) for exp in grouped_expected]

    chi2_stat, chi2_pval = stats.chisquare(grouped_observed, f_exp=grouped_expected)

    plt.subplot(1, 2, 2)
    positions = np.arange(len(grouped_observed))
    plt.bar(positions - 0.15, grouped_observed, width=0.3, label='Actual')
    plt.bar(positions + 0.15, grouped_expected, width=0.3, label='Predicted (Poisson)')
    plt.xticks(positions, [str(i) for i in positions])
    plt.legend()
    plt.title('Poisson Fit Comparison')
    plt.tight_layout()
    plt.show()

    print(f"\nPoisson Distribution Test for Hourly Events:")
    print(f"- Estimated Rate: {poisson_rate:.2f} events per hour")
    print(f"- Chi-Square Test: p-value = {chi2_pval:.4f} → {'Poisson' if chi2_pval > significance_level else 'Not Poisson'}")
    print(f"- Degrees of Freedom: {len(grouped_observed) - 1}")

def run_analysis():
    dataset = import_dataset('babyboom.dat.txt')

    print("=" * 50 + "\nBirth Weight Analysis\n" + "=" * 50)
    analyze_distribution(dataset['WeightAtBirth'], 'All Infants')
    analyze_distribution(dataset[dataset['Gender'] == 1]['WeightAtBirth'], 'Female Infants')
    analyze_distribution(dataset[dataset['Gender'] == 2]['WeightAtBirth'], 'Male Infants')

    print("\n" + "=" * 50 + "\nInter-Event Time Analysis\n" + "=" * 50)
    sorted_dataset = dataset.sort_values('MinutesSinceMidnight')
    time_intervals = sorted_dataset['MinutesSinceMidnight'].diff().dropna()
    verify_exponential(time_intervals)

    print("\n" + "=" * 50 + "\nHourly Events Analysis\n" + "=" * 50)
    dataset['Hour'] = (dataset['MinutesSinceMidnight'] // 60).astype(int)
    hourly_events = dataset['Hour'].value_counts().sort_index()
    test_poisson_distribution(hourly_events)

if __name__ == "__main__":
    run_analysis()
