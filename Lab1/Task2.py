import pandas as pd

# Зчитування даних
data = pd.read_csv("insurance.csv")

print("\n--- Перші 5 рядків ---")
print(data.head())

# а) Основні статистичні характеристики
print("\n--- Статистичні характеристики ---")
print(data.describe())

# б) Чоловіки-курці з витратами > 5000
filtered = data[
    (data["sex"] == "male") &
    (data["smoker"] == "yes") &
    (data["expenses"] > 5000)
]

print("\nКількість чоловіків-курців з витратами > 5000:")
print(len(filtered))

print("\nДані цих людей:")
print(filtered)

# в) Додаємо новий стовпець — витрати на одну людину
data["family_members"] = 1 + data["children"]
data["expense_per_person"] = data["expenses"] / data["family_members"]

print("\n--- DataFrame з новим стовпцем ---")
print(data.head())