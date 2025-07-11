import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


def load_data():
    with open('airportdat.txt', 'r') as file:
        content = file.read().replace('\t', '      ')

    colspecs = [(0, 20), (20, 43), (43, 50), (50, 57), (57, 66), (66, 76), (76, 86)]
    df = pd.read_fwf(StringIO(content), header=None, colspecs=colspecs,
                     names=['Airport', 'City', 'Scheduled_departures', 'Performed_departures',
                            'Enplaned_passengers', 'Enplaned_freight_tons', 'Enplaned_mail_tons'])

    df['Airport'] = df['Airport'].str.strip()
    df['City'] = df['City'].str.strip()


    return df


df = load_data()


def plot_all_data():

    def plot_data(plot_type):
        plt.figure(figsize=(18, 12))
        columns = ['Scheduled_departures', 'Performed_departures', 'Enplaned_passengers',
                   'Enplaned_freight_tons', 'Enplaned_mail_tons']
        positions = [1, 2, 4, 5, 6]

        for col, pos in zip(columns, positions):
            plt.subplot(3, 3, pos)
            if plot_type == 'hist':
                sns.histplot(df[col], kde=True)
            else:
                sns.boxplot(x=df[col])

        plt.tight_layout()
        plt.show()

    plot_data('hist')
    plot_data('box')

    numerical_summary = df[['Scheduled_departures', 'Performed_departures',
                            'Enplaned_passengers', 'Enplaned_freight_tons', 'Enplaned_mail_tons']].describe()
    variance = df[['Scheduled_departures', 'Performed_departures',
                   'Enplaned_passengers', 'Enplaned_freight_tons', 'Enplaned_mail_tons']].var()
    numerical_summary.loc['variance'] = variance

    plt.figure(figsize=(12, 8))
    sns.heatmap(numerical_summary, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Описательная статистика для всех данных')
    plt.show()

    correlation_matrix = df[['Scheduled_departures', 'Performed_departures',
                             'Enplaned_passengers', 'Enplaned_freight_tons', 'Enplaned_mail_tons']].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Попарные коэффициенты корреляции для всех данных')
    plt.show()


def plot_single_column():
    numeric_cols = ['Scheduled_departures', 'Performed_departures',
                    'Enplaned_passengers', 'Enplaned_freight_tons', 'Enplaned_mail_tons']

    print("\nДоступные числовые столбцы для построения графиков:")
    for i, col in enumerate(numeric_cols, 1):
        print(f"{i}. {col}")

    choice = input("Введите номер столбца для построения графиков: ")

    if choice.isdigit() and 1 <= int(choice) <= len(numeric_cols):
        selected_col = numeric_cols[int(choice) - 1]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[selected_col], kde=True, bins=30)
        plt.title(f'Гистограмма: {selected_col}')
        plt.xlabel(selected_col)
        plt.ylabel('Частота')

        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[selected_col])
        plt.title(f'Ящик с усами: {selected_col}')

        plt.tight_layout()
        plt.show()
    else:
        print("Неверный выбор. Пожалуйста, введите номер из списка.")


def print_statistics():
    numeric_cols = ['Scheduled_departures', 'Performed_departures',
                    'Enplaned_passengers', 'Enplaned_freight_tons', 'Enplaned_mail_tons']

    print("\n📋 Описательная статистика:")
    print(df[numeric_cols].describe().round(3))

    print("\n📈 Дисперсия:")
    print(df[numeric_cols].var().round(3))

    print("\n🔗 Матрица корреляций:")
    print(df[numeric_cols].corr().round(3))

    print("\n📌 Предпросмотр данных:")
    print(df.head().to_string(index=False))


def main_menu():
    while True:
        print("\n" + "=" * 50)
        print("АНАЛИЗ ДАННЫХ АЭРОПОРТОВ".center(50))
        print("=" * 50)
        print("1. Показать все графики и статистику")
        print("2. Построить графики для одного столбца")
        print("3. Вывести статистику в консоль")
        print("4. Выход")

        choice = input("Выберите действие (1-4): ")

        if choice == '1':
            plot_all_data()
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