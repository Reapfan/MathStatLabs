import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных
iris = pd.read_csv("iris.txt", sep=",", header=None,
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Диагностика данных
print("Первые 5 строк данных:")
print(iris.head())
print("\nУникальные значения в столбце 'class':", iris["class"].unique())
print("Пропуски в данных:\n", iris.isnull().sum())
print("Форма данных:", iris.shape)

# Удаление строк с пропусками
iris = iris.dropna()

# Проверка классов
expected_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
actual_classes = iris["class"].unique()
if not all(cls in expected_classes for cls in actual_classes):
    print("Предупреждение: Найдены неожиданные значения в столбце 'class':", actual_classes)
    print("Ожидаются:", expected_classes)

# Список характеристик
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Статистические тесты
for feature in features:
    print(f"\nАнализ для {feature}")

    # Группировка по классам
    groups = [group[feature] for name, group in iris.groupby("class")]

    # Проверка на пустые группы
    if any(len(group) == 0 for group in groups):
        print(f"Тесты для {feature} пропущены: одна из групп пуста")
        continue

    # Крускал-Уоллис для равенства распределений
    kw_stat, p_kw = stats.kruskal(*groups)
    print(f"\nКрускал-Уоллис для равенства распределений ({feature}):")
    print(f"  H-статистика = {kw_stat:.2f}, p = {p_kw:.4f} → "
          f"{'распределения одинаковы' if p_kw > 0.05 else 'распределения различаются'}")

    # ANOVA для равенства средних
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"ANOVA для равенства средних ({feature}):")
    print(f"  F-статистика = {f_stat:.2f}, p = {p_anova:.4f} → "
          f"{'средние равны' if p_anova > 0.05 else 'средние различаются'}")

    # Левена для равенства дисперсий
    levene_stat, p_levene = stats.levene(*groups)
    print(f"F-тест Левена для равенства дисперсий ({feature}):")
    print(f"  F-статистика = {levene_stat:.2f}, p = {p_levene:.4f} → "
          f"{'дисперсии равны' if p_levene > 0.05 else 'дисперсии различаются'}")

# Визуализация
plt.figure(figsize=(12, 10))
for i, feature in enumerate(features, 1):
    # Гистограммы
    plt.subplot(4, 2, 2 * i - 1)
    has_data = False
    for cls in expected_classes:
        data = iris[iris["class"] == cls][feature]
        if len(data) > 0:
            sns.histplot(data, label=cls, kde=True, stat='density', alpha=0.6)
            has_data = True
    if has_data:
        plt.title(f'Распределение {feature}')
        plt.xlabel(feature)
        plt.ylabel('Плотность')
        plt.legend()
    else:
        plt.title(f'Распределение {feature} (нет данных)')
        plt.xlabel(feature)
        plt.ylabel('Плотность')

    # Boxplot
    plt.subplot(4, 2, 2 * i)
    sns.boxplot(x='class', y=feature, data=iris, hue='class', palette='Set2', legend=False)
    plt.title(f'Сравнение {feature} по классам')
    plt.xlabel('Класс')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()
