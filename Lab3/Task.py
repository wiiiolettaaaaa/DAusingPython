import pandas as pd
import numpy as np

df = pd.read_csv('merc.csv')

print("--- Завдання 1: Інформація про DataFrame ---")
df.info()
print("\n")

print("--- Завдання 2 ---")

# а)
cols = df.columns.tolist()
print("а) Назви стовпців:\n", cols, "\n")

# б)
model_counts = df['model'].value_counts()
print("б) Кількість машин кожної моделі (перші 5):\n", model_counts.head(), "\n")

# в)
random_car = df[(df['fuelType'] == 'Diesel') &
                (df['transmission'] == 'Automatic') &
                (df['price'] < 15000)].sample(n=1)
print("в) Випадкова машина за умовами:\n", random_car, "\n")

# г)
new_row = {'model': ' A Class', 'year': 2023, 'price': 30000,
           'transmission': 'Automatic', 'mileage': 5000,
           'fuelType': 'Petrol', 'tax': 150, 'mpg': 45.0, 'engineSize': 2.0}

# Використовуємо loc для додавання рядка в кінець DataFrame
df.loc[len(df)] = new_row
print("г) Новий рядок успішно додано. Загальна кількість рядків тепер:", len(df), "\n")


print("--- Завдання 3 ---")

# a)
max_mpg_by_fuel = df.groupby('fuelType')['mpg'].max()
print("а) Максимальний mpg за типом пального:\n", max_mpg_by_fuel, "\n")

# б)
df['avg_price_for_year'] = df.groupby('year')['price'].transform('mean')
print("б) Стовпець avg_price_for_year додано. Фрагмент даних:\n", df[['year', 'price', 'avg_price_for_year']].head(), "\n")

# в)
high_mileage_years_df = df.groupby('year').filter(lambda x: x['mileage'].max() > 100000)
print("в) Машини з років, де максимальний пробіг > 100 000 (перші 3 рядки):\n", high_mileage_years_df.head(), "\n")


print("--- Завдання 4 ---")

tax_pivot = df.pivot_table(values='tax', index='year', columns='fuelType', aggfunc='mean')
print("Зведена таблиця середніх податків:\n", tax_pivot.tail(), "\n") # Виводжу останні рядки для наочності

tax_2017_diesel = tax_pivot.loc[2017, 'Diesel']
print(f"Середній податок на машини 2017 року (Diesel): {tax_2017_diesel:.2f}")