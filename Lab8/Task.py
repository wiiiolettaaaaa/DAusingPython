import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

print("=== Етап 1: Підготовка даних ===")
df = pd.read_csv('glass.csv')

df['is_building_glass'] = df['Type'].apply(lambda x: 1 if x in [1, 2] else 0)

X = df.drop(columns=['Type', 'is_building_glass'])
y = df['is_building_glass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Розмірність навчальної вибірки: {X_train.shape}")
print(f"Розмірність тестової вибірки: {X_test.shape}\n")

print("=== Етап 2: Базові моделі ===")
models = {
    'Логістична регресія (Logistic Regression)': LogisticRegression(random_state=42),
    'Випадковий ліс (Random Forest)': RandomForestClassifier(random_state=42),
    'Метод опорних векторів (SVC)': SVC(probability=True, random_state=42)
}


def evaluate_model(model, X_train_data, X_test_data, y_train, y_test, model_name):
    model.fit(X_train_data, y_train)
    y_pred = model.predict(X_test_data)
    y_proba = model.predict_proba(X_test_data)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"--- {model_name} ---")
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    return model


for name, model in models.items():
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
print("\n")

print("=== Етап 3.а: Підбір гіперпараметрів (Random Forest) ===")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

print(f"Найкращі гіперпараметри: {grid_search.best_params_}")
evaluate_model(best_rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest (Після GridSearchCV)")
print("\n")

print("=== Етап 3.б: Вибір ознак (SelectKBest) ===")
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()].tolist()
print(f"Вибрані ознаки: {selected_features}")

lr_selected = LogisticRegression(random_state=42)
evaluate_model(lr_selected, X_train_selected, X_test_selected, y_train, y_test,
               "Логістична регресія (Тільки вибрані ознаки)")
print("\n")

print("=== Етап 3.в: Зменшення розмірності за допомогою PCA ===")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Кількість компонент після PCA: {pca.n_components_} (з початкових {X_train.shape[1]})")
print(f"Загальна пояснена дисперсія: {sum(pca.explained_variance_ratio_):.4f}")

svc_pca = SVC(probability=True, random_state=42)
evaluate_model(svc_pca, X_train_pca, X_test_pca, y_train, y_test, "Метод опорних векторів (Після PCA)")