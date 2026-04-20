import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv('merc.csv')

fig, axes = plt.subplots(3, 1, figsize=(14, 18))

sns.countplot(data=df, x='year', ax=axes[0], hue='year', palette='Blues_d', legend=False)
axes[0].set_title('а) Кількість автомобілів кожного року реєстрації')
axes[0].set_xlabel('Рік реєстрації')
axes[0].set_ylabel('Кількість')

max_mileage = df.groupby('year')['mileage'].max().reset_index()
sns.barplot(data=max_mileage, x='year', y='mileage', ax=axes[1], hue='year', palette='Reds_d', legend=False)
axes[1].set_title('б) Максимальний пробіг автомобілів за роком реєстрації')
axes[1].set_xlabel('Рік реєстрації')
axes[1].set_ylabel('Максимальний пробіг')

sns.barplot(data=df, x='year', y='mileage', hue='fuelType', estimator='mean', errorbar=None, ax=axes[2])
axes[2].set_title('в) Середній пробіг за роками реєстрації та типом палива')
axes[2].set_xlabel('Рік реєстрації')
axes[2].set_ylabel('Середній пробіг')
axes[2].legend(title='Тип палива')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(df['mpg'], bins=40, kde=True, ax=axes[0], color='teal')
axes[0].set_title('Загальний розподіл витрат палива (mpg)')
axes[0].set_xlabel('Миль на галон (mpg)')

sns.histplot(data=df, x='mpg', hue='fuelType', bins=40, kde=True, element='step', ax=axes[1])
axes[1].set_title('Розподіл витрат палива за типом палива')
axes[1].set_xlabel('Миль на галон (mpg)')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(y=df['mileage'], ax=axes[0], color='skyblue')
axes[0].set_title('Загальна діаграма розмаху пробігу')
axes[0].set_ylabel('Пробіг')

sns.boxplot(data=df, x='transmission', y='mileage', ax=axes[1], hue='transmission', palette='Set2', legend=False)
axes[1].set_title('Діаграма розмаху пробігу за типом коробки передач')
axes[1].set_xlabel('Коробка передач')
axes[1].set_ylabel('Пробіг')

plt.tight_layout()
plt.show()

corr_year_mileage = df['year'].corr(df['mileage'])
corr_mpg_price = df['mpg'].corr(df['price'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x='year', y='mileage', alpha=0.3, ax=axes[0], color='purple')
axes[0].set_title(f'а) Пробіг vs Рік реєстрації\n(Кореляція: {corr_year_mileage:.2f})')
axes[0].set_xlabel('Рік реєстрації')
axes[0].set_ylabel('Пробіг')

sns.scatterplot(data=df, x='mpg', y='price', alpha=0.3, ax=axes[1], color='orange')
axes[1].set_title(f'б) Ціна vs Витрати палива (mpg)\n(Кореляція: {corr_mpg_price:.2f})')
axes[1].set_xlabel('Миль на галон (mpg)')
axes[1].set_ylabel('Ціна')

plt.tight_layout()
plt.show()

print(f"Коефіцієнт кореляції Пірсона (Рік реєстрації та Пробіг): {corr_year_mileage:.3f}")
print(f"Коефіцієнт кореляції Пірсона (Ціна та Витрати палива - mpg): {corr_mpg_price:.3f}")