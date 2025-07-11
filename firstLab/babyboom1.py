import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')

def load_data():
    try:
        col_widths = [
            (1, 8),
            (9, 16),
            (17, 24),
            (25, 32)
        ]

        data = []
        with open('babyboom.dat.txt', 'r') as file:
            for line in file:
                row = []
                for start, end in col_widths:
                    value = line[start - 1:end].strip()
                    row.append(value)
                data.append(row)

        columns = ['Time', 'Sex', 'Birth_weight', 'Minutes_after_midnight']
        df = pd.DataFrame(data, columns=columns)

        numeric_cols = ['Birth_weight', 'Minutes_after_midnight']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].str.replace(' ', ''), errors='coerce')

    except Exception as e:
        print(f"Первый метод чтения не сработал: {e}. Пробуем второй метод...")
        with open('babyboom.dat.txt', 'r') as f:
            df = pd.read_fwf(StringIO(f.read()), header=None,
                             names=["Time", "Sex", "Birth_weight", "Minutes_after_midnight"])

    return df


df = load_data()

def print_statistics():
    numeric_cols = ['Birth_weight', 'Minutes_after_midnight']

    print("\n📋 Описательная статистика:")
    print(df[numeric_cols].describe().round(3))

    print("\n📈 Дисперсия:")
    print(df[numeric_cols].var().round(3))

    print("\n🔗 Матрица корреляций:")
    print(df[numeric_cols].corr().round(3))


def plot_single_column():
    numeric_cols = ['Birth_weight', 'Minutes_after_midnight']
    col_names = ['Вес при рождении (г)', 'Минут после полуночи']

    print("\nДоступные числовые столбцы для построения графиков:")
    for i, (col, name) in enumerate(zip(numeric_cols, col_names), 1):
        print(f"{i}. {name}")

    choice = input("Введите номер столбца для построения графиков: ")

    if choice.isdigit() and 1 <= int(choice) <= len(numeric_cols):
        selected_col = numeric_cols[int(choice) - 1]
        selected_name = col_names[int(choice) - 1]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[selected_col], kde=True, bins=15)
        plt.title(f'Гистограмма: {selected_name}')
        plt.xlabel(selected_name)
        plt.ylabel('Частота')

        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[selected_col])
        plt.title(f'Ящик с усами: {selected_name}')

        plt.tight_layout()
        plt.show()
    else:
        print("Неверный выбор. Пожалуйста, введите номер из списка.")


def plot_distributions(plot_type='hist'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, col in enumerate(['Birth_weight', 'Minutes_after_midnight']):
        title = 'веса при рождении (г)' if i == 0 else 'времени рождения'

        if plot_type == 'hist':
            sns.histplot(df[col], kde=True, ax=ax[i], bins=15)
        else:
            sns.boxplot(x=df[col], ax=ax[i])

        ax[i].set_title(f'{title.capitalize()}')
        ax[i].set_xlabel('Вес (г)' if i == 0 else 'Минуты после полуночи')
        if plot_type == 'hist': ax[i].set_ylabel('Частота')

    plt.suptitle(f'Распределение {plot_type.replace("hist", "гистограмма").replace("box", "ящик с усами")}')
    plt.tight_layout()
    plt.show()


def analyze_data():
    numeric_df = df[['Birth_weight', 'Minutes_after_midnight']].apply(pd.to_numeric, errors='coerce').dropna()

    stats = numeric_df.describe()
    stats.loc['variance'] = numeric_df.var()

    plt.figure(figsize=(10, 4))
    sns.heatmap(stats, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Описательная статистика')
    plt.tight_layout()
    plt.show()

    corr = numeric_df.corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.show()

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()

    outliers_df = pd.DataFrame({
        'variable': outliers.index,
        'outliers': outliers.values
    })

    plt.figure(figsize=(8, 4))
    sns.barplot(x='variable', y='outliers', hue='variable', data=outliers_df,
                legend=False)
    plt.title('Количество выбросов')
    plt.xlabel('Показатель')
    plt.ylabel('Количество выбросов')
    plt.tight_layout()
    plt.show()

def main_menu():
    while True:
        print("\n" + "=" * 50)
        print("АНАЛИЗ ДАННЫХ О НОВОРОЖДЕННЫХ".center(50))
        print("=" * 50)
        print("1. Показать все графики и статистику")
        print("2. Построить графики для одного показателя")
        print("3. Вывести статистику в консоль")
        print("4. Выход")

        choice = input("Выберите действие (1-4): ")

        if choice == '1':
            plot_distributions('hist')
            plot_distributions('box')
            analyze_data()
        elif choice == '2':
            plot_single_column()
        elif choice == '3':
            print_statistics()
        elif choice == '4':
            print("Выход из программы...")
            break
        else:
            print("Неверный выбор. Пожалуйста, введите число от 1 до 4.")


if __name__ == "__main__":
    main_menu()
