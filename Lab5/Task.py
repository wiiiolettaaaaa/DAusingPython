import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('Microsoft_Stock.csv', parse_dates=['Date'], index_col='Date')

df = df.sort_index()

plt.figure(figsize=(10, 5))
df['Close'].plot(title='1а. Загальний графік ціни закриття (Close)', color='blue')
plt.ylabel('Ціна ($)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
df.loc['2019', 'Close'].plot(title='1б. Ціна закриття за 2019 рік', color='green')
plt.ylabel('Ціна ($)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
df.loc['2018-09', 'Close'].plot(title='1в. Ціна закриття за вересень 2018', marker='o', color='red')
plt.ylabel('Ціна ($)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
df.loc['2015-11':'2018-01', 'Close'].plot(title='1г. Ціна закриття (Листопад 2015 - Січень 2018)', color='purple')
plt.ylabel('Ціна ($)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
close_2017 = df.loc['2017', 'Close'].copy()
close_2018 = df.loc['2018', 'Close'].copy()

close_2017.index = close_2017.index.map(lambda t: t.replace(year=2000))
close_2018.index = close_2018.index.map(lambda t: t.replace(year=2000))

close_2017.plot(label='2017 рік', color='orange')
close_2018.plot(label='2018 рік', color='teal')

plt.title('1д. Порівняння ціни закриття за 2017 та 2018 роки')
plt.ylabel('Ціна ($)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.legend()
plt.grid(True)
plt.show()

mean_high_2016 = df.loc['2016', 'High'].mean()
print(f"2а. Середнє значення найбільшої ціни за день у 2016 році: ${mean_high_2016:.2f}")

mean_high_monthly = df['High'].resample('M').mean()
print("\n2б. Середнє значення найбільшої ціни (перші 5 місяців для прикладу):")
print(mean_high_monthly.head())

mean_high_q1_2019_weekly = df.loc['2019-01-01':'2019-03-31', 'High'].resample('W').mean()
print("\n2в. Середнє щотижневе 'High' за Q1 2019:")
print(mean_high_q1_2019_weekly)

plt.figure(figsize=(10, 5))

summer_2019_pct = df.loc['2019-06-01':'2019-08-31', 'High'].pct_change() * 100

summer_2019_pct.plot(title='2г. Щоденна зміна найбільшої ціни у відсотках (Літо 2019)', color='brown', marker='.')
plt.ylabel('Зміна (%)')
plt.axhline(0, color='black', linestyle='--', linewidth=1) # Лінія нуля
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sep_2017_high = df.loc['2017-09', 'High']
sep_2017_rolling = sep_2017_high.rolling(window=5).mean()

sep_2017_high.plot(label='Оригінальна ціна High', marker='o', linestyle='--', alpha=0.6)
sep_2017_rolling.plot(label='Ковзне середнє (вікно=5)', color='red', linewidth=2)

plt.title('2д. Ковзне середнє ціни High за Вересень 2017')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid(True)
plt.show()