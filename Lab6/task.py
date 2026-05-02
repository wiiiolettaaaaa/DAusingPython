import pandas as pd
import numpy as np

df = pd.read_json('Version 9.json', orient='index')

print("--- Перші 5 рядків до обробки ---")
print(df.head())

df.rename(columns={
    'species': 'Вид',
    'island': 'Острів',
    'culmen_length_mm': 'Довжина_дзьоба_мм',
    'culmen_depth_mm': 'Глибина_дзьоба_мм',
    'flipper_length_mm': 'Довжина_ласта_мм',
    'body_mass_g': 'Маса_тіла_г',
    'sex': 'Стать'
}, inplace=True)

df['Вид'] = df['Вид'].str.replace('&', '', regex=False).str.title()

df['Стать'] = df['Стать'].replace({
    'MALE': 'Чоловіча',
    'FEMALE': 'Жіноча',
    'girl': 'Жіноча'
})

df['Стать'] = df['Стать'].apply(lambda x: x if x in ['Чоловіча', 'Жіноча'] else np.nan)


df.dropna(subset=['Довжина_дзьоба_мм', 'Глибина_дзьоба_мм', 'Довжина_ласта_мм', 'Маса_тіла_г'], how='all', inplace=True)

df.loc[df['Довжина_дзьоба_мм'] > 1000, 'Довжина_дзьоба_мм'] = np.nan
df.loc[df['Глибина_дзьоба_мм'] > 1000, 'Глибина_дзьоба_мм'] = np.nan

числові_стовпці = ['Довжина_дзьоба_мм', 'Глибина_дзьоба_мм', 'Довжина_ласта_мм', 'Маса_тіла_г']
for col in числові_стовпці:
    df[col] = df[col].fillna(df[col].median())

df['Стать'] = df['Стать'].fillna('Невідомо')

df.drop_duplicates(inplace=True)

print("\n--- Перші 5 рядків після обробки ---")
print(df.head())

print("\n--- Інформація про очищений датафрейм ---")
df.info()

print("\n--- Статистичні характеристики числових даних ---")
print(df.describe())

print("\n--- Статистичні характеристики категоріальних даних ---")
print(df.describe(include=['object']))