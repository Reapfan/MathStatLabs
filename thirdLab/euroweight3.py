import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных
df = pd.read_csv("euroweight.dat.txt", sep="\t", header=None, names=["ID", "weight", "batch"])

# Функция для проверки нормальности
def is_normal(weights: pd.Series, label: str = '') -> None:
    shapiro_stat, shapiro_p = stats.shapiro(weights)
    print(f"\n{label} – вес монет:")
    print(f"  Шапиро-Уилк: p = {shapiro_p:.4f} → "
          f"{'нормальность не отвергается' if shapiro_p > 0.05 else 'нормальность отвергается'}")

# Проверка нормальности для общей выборки
weights_all = df["weight"]
is_normal(weights_all, 'Общая выборка')

# Проверка нормальности для каждой упаковки
for batch, group in df.groupby("batch"):
    wei = group["weight"]
    is_normal(wei, f'Упаковка {batch}')

# ANOVA для проверки равенства средних
batches = [group["weight"] for name, group in df.groupby("batch")]
f_stat, p_anova = stats.f_oneway(*batches)
print(f"\nANOVA для равенства средних весов по упаковкам:")
print(f"  F-статистика = {f_stat:.2f}, p = {p_anova:.4f} → "
      f"{'средние равны' if p_anova > 0.05 else 'средние различаются'}")

# Визуализация: гистограммы и boxplot
plt.figure(figsize=(12, 5))

# Гистограмма для веса монет
plt.subplot(1, 2, 1)
sns.histplot(weights_all, color='gray', label='Все монеты', kde=True, stat='density')
for batch, group in df.groupby("batch"):
    sns.histplot(group["weight"], label=f'Упаковка {batch}', kde=True, stat='density', alpha=0.5)
plt.title('Распределение веса монет')
plt.xlabel('Вес (г)')
plt.ylabel('Плотность')
plt.legend()

# Boxplot для сравнения веса по упаковкам
plt.subplot(1, 2, 2)
sns.boxplot(x='batch', y='weight', data=df, hue='batch', palette='Set2', legend=False)
plt.title('Сравнение веса монет по упаковкам')
plt.xlabel('Упаковка')
plt.ylabel('Вес (г)')

plt.tight_layout()
plt.show()
