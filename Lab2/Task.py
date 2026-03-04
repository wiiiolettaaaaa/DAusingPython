import pandas as pd
from scipy import stats

df = pd.read_csv('frogs.csv')

frogs_observed = df[df['pres.abs'] == 0]
std_meanmin = frogs_observed['meanmin'].std()
print(f"Середньоквадратичне відхилення мінімальних температур (де є жаби): {std_meanmin:.4f}\n")

stat_shapiro, p_shapiro = stats.shapiro(df['distance'])
print(f"Тест Шапіро-Вілка для 'distance': Статистика = {stat_shapiro:.4f}, p-value = {p_shapiro:.4e}")
if p_shapiro < 0.05:
    print("Висновок: p-value < 0.05, отже відхиляємо основну гіпотезу. Розподіл відстані до поселення НЕ є нормальним.\n")
else:
    print("Висновок: p-value >= 0.05, немає підстав відхилити основну гіпотезу. Розподіл можна вважати нормальним.\n")

corr_pearson, p_pearson = stats.pearsonr(df['altitude'], df['avrain'])
print(f"Кореляція Пірсона між 'altitude' та 'avrain': r = {corr_pearson:.4f}, p-value = {p_pearson:.4e}")
if p_pearson < 0.05:
    print("Висновок: p-value < 0.05, статистично значущий зв'язок існує.")
    if abs(corr_pearson) > 0.7:
        print("Зв'язок сильний.\n")
    elif abs(corr_pearson) > 0.3:
        print("Зв'язок середній.\n")
    else:
        print("Зв'язок слабкий.\n")
else:
    print("Висновок: p-value >= 0.05, статистично значущого зв'язку немає.\n")

avrain_frogs = df[df['pres.abs'] == 0]['avrain']
avrain_no_frogs = df[df['pres.abs'] == 1]['avrain']


t_stat, p_ttest = stats.ttest_ind(avrain_frogs, avrain_no_frogs)
print(f"T-тест для незалежних вибірок: t-статистика = {t_stat:.4f}, p-value = {p_ttest:.4e}")

if p_ttest < 0.05:
    print("Висновок: p-value < 0.05, відхиляємо основну гіпотезу H0. Середня кількість опадів СТАТИСТИЧНО ВІДРІЗНЯЄТЬСЯ в місцях, де є жаби, і де їх немає.")
else:
    print("Висновок: p-value >= 0.05, приймаємо основну гіпотезу H0. Відмінність у середній кількості опадів статистично НЕзначуща.")