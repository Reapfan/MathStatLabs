import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Определение пути к файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "surgery.xlsx")
print(f"Попытка открыть файл: {file_path}")

# Чтение данных, пропуская первую строку (заголовок)
try:
    df = pd.read_excel(file_path, skiprows=1)
except FileNotFoundError:
    print(f"Ошибка: Файл 'surgery.xlsx' не найден по пути: {file_path}")
    print(f"Текущий рабочий каталог: {os.getcwd()}")
    print("Проверьте, что файл находится в той же папке, что и скрипт, и правильно написан.")
    exit(1)

# Переименование столбцов
df.columns = ["V_left_before", "V_right_before", "V_left_after", "V_right_after"]

# Диагностика данных
print("\nПервые 5 строк данных:")
print(df.head())
print("\nПропуски в данных:\n", df.isnull().sum())
print("Форма данных:", df.shape)

# Проверка успешности операции
df.loc[:, "success"] = (df["V_right_after"] > df["V_right_before"]) & (df["V_left_after"] > df["V_left_before"])
num_successes = df["success"].sum()
n = len(df)
success_rate = num_successes / n

print(f"\nАнализ успешности операции:")
print(f"Количество пациентов: {n}")
print(f"Успешных операций: {num_successes}")
print(f"Доля успешных операций: {success_rate:.4f}")

# Биномиальный тест
for p0 in [0.7, 0.8]:
    p_value = stats.binomtest(num_successes, n, p=p0, alternative='greater').pvalue
    print(f"\nПроверка против p0 = {p0}:")
    print(f"  p-значение = {p_value:.6f} → "
          f"{'операция статистически успешна' if p_value < 0.05 else 'нет статистических оснований считать операцию успешной'}")

# Визуализация (на основе данных без пропусков)
df_clean = df.dropna()
print("\nФорма данных после удаления пропусков:", df_clean.shape)

if len(df_clean) > 0:
    plt.figure(figsize=(12, 8))

    # Гистограммы для V left
    plt.subplot(2, 2, 1)
    sns.histplot(df_clean["V_left_before"], label="До операции, V left", kde=True, stat='density', alpha=0.5, color='blue')
    sns.histplot(df_clean["V_left_after"], label="После операции, V left", kde=True, stat='density', alpha=0.5, color='orange')
    plt.title("Распределение V left")
    plt.xlabel("V left")
    plt.ylabel("Плотность")
    plt.legend()

    # Гистограммы для V right
    plt.subplot(2, 2, 2)
    sns.histplot(df_clean["V_right_before"], label="До операции, V right", kde=True, stat='density', alpha=0.5, color='blue')
    sns.histplot(df_clean["V_right_after"], label="После операции, V right", kde=True, stat='density', alpha=0.5, color='orange')
    plt.title("Распределение V right")
    plt.xlabel("V right")
    plt.ylabel("Плотность")
    plt.legend()

    # Boxplot для V left
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_clean[["V_left_before", "V_left_after"]], palette=['blue', 'orange'])
    plt.xticks([0, 1], ["До операции, V left", "После операции, V left"])
    plt.title("Сравнение V left")
    plt.ylabel("V left")

    # Boxplot для V right
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_clean[["V_right_before", "V_right_after"]], palette=['blue', 'orange'])
    plt.xticks([0, 1], ["До операции, V right", "После операции, V right"])
    plt.title("Сравнение V right")
    plt.ylabel("V right")

    plt.tight_layout()
    plt.show()
else:
    print("Визуализация пропущена: нет данных")