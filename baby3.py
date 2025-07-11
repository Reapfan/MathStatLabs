import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных с фиксированной шириной столбцов
col_widths = [
    (1, 8),   # Time of birth
    (9, 16),  # Sex
    (17, 24), # Birth weight in grams
    (25, 32)  # Number of minutes after midnight
]

data = []
with open('babyboom.dat.txt', 'r') as file:
    for line in file:
        data.append([line[s - 1:e].strip() for s, e in col_widths])

# Создание DataFrame
columns = ['Time of birth', 'Sex', 'Birth weight in grams', 'Number of minutes after midnight']
df = pd.DataFrame(data, columns=columns)

# Преобразование числовых столбцов
numeric_cols = ['Birth weight in grams', 'Number of minutes after midnight']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].str.replace(' ', ''), errors='coerce')

# Разделение данных по полу
weights_all = df['Birth weight in grams']
boys = df[df['Sex'] == '2']['Birth weight in grams']
girls = df[df['Sex'] == '1']['Birth weight in grams']

# T-тест для проверки равенства средних
t_stat, p_ttest = stats.ttest_ind(boys, girls, equal_var=False)
print(f"\nT-тест для равенства средних весов (мальчики vs девочки):")
print(f"  t-статистика = {t_stat:.2f}, p = {p_ttest:.4f} → "
      f"{'средние равны' if p_ttest > 0.05 else 'средние различаются'}")

# F-тест Левена для проверки равенства дисперсий
levene_stat, p_levene = stats.levene(boys, girls)
print(f"\nF-тест Левена для равенства дисперсий (мальчики vs девочки):")
print(f"  F-статистика = {levene_stat:.2f}, p = {p_levene:.4f} → "
      f"{'дисперсии равны' if p_levene > 0.05 else 'дисперсии различаются'}")

# Визуализация: гистограммы и boxplot
plt.figure(figsize=(12, 5))

# Гистограмма для веса мальчиков и девочек
plt.subplot(1, 2, 1)
sns.histplot(boys, color='blue', label='Мальчики', kde=True, stat='density')
sns.histplot(girls, color='pink', label='Девочки', kde=True, stat='density')
plt.title('Распределение веса при рождении')
plt.xlabel('Вес (г)')
plt.ylabel('Плотность')
plt.legend()

# Boxplot для сравнения веса
plt.subplot(1, 2, 2)
sns.boxplot(x='Sex', y='Birth weight in grams', hue='Sex', data=df, palette={'1': 'pink', '2': 'blue'}, legend=False)
plt.title('Сравнение веса мальчиков и девочек')
plt.xlabel('Пол (1 = девочки, 2 = мальчики)')
plt.ylabel('Вес (г)')

plt.tight_layout()
plt.show()