import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = sns.load_dataset('mpg')

df = df.dropna()

print("--- ЧАСТИНА 1: РЕГРЕСІЯ ---")
X = df[['displacement', 'weight', 'cylinders']]
y = df['horsepower']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

lr_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('lr', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

print("\nМодель 1: Лінійна регресія")
print(f"R^2: {r2_score(y_test, lr_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_test, lr_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.4f}")

rf_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=0))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

print("\nМодель 2: Випадковий ліс")
print(f"R^2: {r2_score(y_test, rf_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_test, rf_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.4f}")

print("\nПокращення моделі Random Forest за допомогою GridSearchCV...")
search_space = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 5, 10]
}

grid = GridSearchCV(rf_pipeline, search_space, cv=5, scoring='r2')
grid.fit(X_train, y_train)

print(f"Найкращі гіперпараметри: {grid.best_params_}")
print(f"Найкраща оцінка R^2 при крос-валідації: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
best_preds = best_model.predict(X_test)
print(f"R^2 покращеної моделі на тестових даних: {r2_score(y_test, best_preds):.4f}")

print("\n\n--- ЧАСТИНА 2: КЛАСТЕРИЗАЦІЯ ---")
kmeans_data = df[['horsepower', 'mpg']]

scaler = StandardScaler()
kmeans_data_scaled = scaler.fit_transform(kmeans_data)

d_list = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init='auto')
    kmeans.fit(kmeans_data_scaled)
    d_list.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), d_list, marker='o')
plt.xlabel('Кількість кластерів К')
plt.ylabel('Cума квадратів відстаней в середині кластера')
plt.title('Метод ліктя для визначення оптимальної кількості кластерів')
plt.grid(True)
plt.show()

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=0, n_init='auto')
cluster_labels = kmeans_final.fit_predict(kmeans_data_scaled)

sil_score = silhouette_score(kmeans_data_scaled, cluster_labels)
db_score = davies_bouldin_score(kmeans_data_scaled, cluster_labels)

print(f"Обрана кількість кластерів: {optimal_k}")
print(f"Коефіцієнт силуету (ближче до 1 - краще): {sil_score:.4f}")
print(f"Оцінка Девіса-Болдіна (ближче до 0 - краще): {db_score:.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=kmeans_data['horsepower'], y=kmeans_data['mpg'], hue=cluster_labels, palette='viridis')
plt.title(f'Кластеризація автомобілів (K-Means, k={optimal_k})')
plt.xlabel('Кінські сили (horsepower)')
plt.ylabel('Милі на галон (mpg)')
plt.show()