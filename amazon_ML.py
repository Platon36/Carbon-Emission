import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'Carbon Emission.csv'
data = pd.read_csv(file_path)
import seaborn as sns

# Разделение данных
X = data.drop('CarbonEmission', axis=1)
y = data['CarbonEmission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Предобработка числовых данных
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = SimpleImputer(strategy='median')
X_train_numeric = numeric_transformer.fit_transform(X_train[numeric_features])
X_test_numeric = numeric_transformer.transform(X_test[numeric_features])

# Масштабирование числовых данных
scaler = StandardScaler()
X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
X_test_numeric_scaled = scaler.transform(X_test_numeric)

# Предобработка категориальных данных
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = SimpleImputer(strategy='constant', fill_value='missing')
X_train_categorical = categorical_transformer.fit_transform(X_train[categorical_features])
X_test_categorical = categorical_transformer.transform(X_test[categorical_features])

# Преобразование категориальных данных в one-hot кодировку
onehot = OneHotEncoder(handle_unknown='ignore')
X_train_categorical_onehot = onehot.fit_transform(X_train_categorical).toarray()
X_test_categorical_onehot = onehot.transform(X_test_categorical).toarray()

# Объединение предобработанных числовых и категориальных данных
X_train_preprocessed = np.hstack((X_train_numeric_scaled, X_train_categorical_onehot))
X_test_preprocessed = np.hstack((X_test_numeric_scaled, X_test_categorical_onehot))

#print(X_test.info())


# Множественная регрессия
lr_model = LinearRegression()
lr_model.fit(X_train_preprocessed, y_train)
lr_train_score = lr_model.score(X_train_preprocessed, y_train)
lr_test_score = lr_model.score(X_test_preprocessed, y_test)

print("Множественная регрессия:")
print(f"Точность на обучающей выборке: {lr_train_score}")
print(f"Точность на тестовой выборке: {lr_test_score}")


# Случайный лес
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train_preprocessed, y_train)
rf_train_score = rf_model.score(X_train_preprocessed, y_train)
rf_test_score = rf_model.score(X_test_preprocessed, y_test)

print("\nСлучайный лес:")
print(f"Точность на обучающей выборке: {rf_train_score}")
print(f"Точность на тестовой выборке: {rf_test_score}")


from sklearn.model_selection import GridSearchCV
# Обучение методом дерева решений
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_preprocessed, y_train)
dt_train_score = dt_model.score(X_train_preprocessed, y_train)
dt_test_score = dt_model.score(X_test_preprocessed, y_test)

print("\nДерево решений:")
print(f"Точность на обучающей выборке: {dt_train_score}")
print(f"Точность на тестовой выборке: {dt_test_score}")

# # Имена числовых признаков
# numeric_feature_names = list(numeric_features)
#
# # Имена категориальных признаков после OneHotEncoding
# categorical_feature_names = list(onehot.get_feature_names_out(categorical_features))
#
# # Объединение списков числовых и категориальных имен признаков
# all_feature_names = numeric_feature_names + categorical_feature_names
#
# plt.figure(figsize=(20,13))
# plot_tree(dt_model, filled=True, feature_names=all_feature_names, max_depth=3)
# plt.show()



# # Определение параметров для перебора
# param_grid = {
#     'max_depth': [None, 10, 20, 30],
#     'n_estimators': [10, 50, 100, 200]
# }
#
# # Создание экземпляра модели случайного леса
# rf_model = RandomForestRegressor(random_state=42)
#
# # Создание экземпляра объекта GridSearchCV
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
#
# # Обучение модели на сетке параметров
# grid_search.fit(X_train_preprocessed, y_train)
#
# # Вывод лучших параметров
# print("Лучшие параметры:", grid_search.best_params_)
#
# # Оценка модели с лучшими параметрами
# best_rf_model = grid_search.best_estimator_
# best_rf_train_score = best_rf_model.score(X_train_preprocessed, y_train)
# best_rf_test_score = best_rf_model.score(X_test_preprocessed, y_test)
#
# print("\nСлучайный лес с подобранными параметрами:")
# print(f"Точность на обучающей выборке: {best_rf_train_score}")
# print(f"Точность на тестовой выборке: {best_rf_test_score}")

