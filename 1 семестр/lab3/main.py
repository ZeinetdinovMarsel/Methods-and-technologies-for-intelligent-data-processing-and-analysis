import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')


class RegressionAnalyzer:
    def __init__(self):
        self.results = {}
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(
                n_estimators=50,
                max_depth=7,
                min_samples_split=50,
                random_state=42,
                n_jobs=-1
            ),
            'DecisionTree': DecisionTreeRegressor(max_depth=7, random_state=42),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5)
        }

    def load_data(self):

        bottle_data = pd.read_csv('bottle.csv', low_memory=False, nrows=500000)
        cast_data = pd.read_csv('cast.csv', low_memory=False)

        print(f"Загружено строк из bottle: {bottle_data.shape[0]}")
        print(f"Загружено строк из cast: {cast_data.shape[0]}")

        merged_data = bottle_data.merge(cast_data, on='Cst_Cnt', how='left', suffixes=('_bottle', '_cast'))
        print(f"Объединенный датасет: {merged_data.shape}")

        return merged_data

    def prepare_features(self, data, target_column='T_degC'):
        if target_column not in data.columns:
            return None, None, None, None

        potential_leakage_columns = []
        for col in data.columns:
            if col.endswith('_bottle') and target_column in col:
                potential_leakage_columns.append(col)
            if col in ['R_TEMP', 'R_POTEMP']:
                potential_leakage_columns.append(col)

        potential_leakage_columns = [col for col in potential_leakage_columns if col != target_column]
        features_df = data.drop(columns=potential_leakage_columns, errors='ignore')

        numeric_data = features_df.select_dtypes(include=[np.number])
        features = numeric_data.drop(columns=[target_column], errors='ignore')

        valid_mask = data[target_column].notna()
        features = features[valid_mask]
        target = data.loc[valid_mask, target_column]

        non_empty_features = features.columns[features.notna().any()].tolist()
        features = features[non_empty_features]

        if target_column in features.columns:
            features = features.drop(columns=[target_column])

        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(features)
        features_imputed = pd.DataFrame(features_imputed, columns=features.columns)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)
        features_final = pd.DataFrame(features_scaled, columns=features.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            features_final, target, test_size=0.3, random_state=42, shuffle=True
        )

        print(f"Обучающая выборка: {X_train.shape}")
        print(f"Тестовая выборка: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)

            metrics = {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred),
                'Explained_Variance': explained_variance_score(y_test, y_pred),
                'Train_Time': train_time
            }

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            metrics['CV_R2_Mean'] = cv_scores.mean()
            metrics['CV_R2_Std'] = cv_scores.std()

            self.results[model_name] = metrics

            print(f"\n{model_name}:")
            print(f"R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f}")
            print(f"Время обучения: {metrics['Train_Time']:.2f}с")
            print(f"Кросс-валидация R²: {metrics['CV_R2_Mean']:.4f} (±{metrics['CV_R2_Std']:.4f})")

            return metrics, y_pred

        except Exception as e:
            print(f"Ошибка в {model_name}: {str(e)}")
            return None, None

    def plot_predictions(self, y_true, y_pred, model_name):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Истинные значения')
        plt.ylabel('Предсказания')
        plt.title(f'{model_name}\nПредсказания vs Истинные значения')

        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Предсказания')
        plt.ylabel('Остатки')
        plt.title('Диаграмма остатков')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model, feature_names, model_name, top_n=10):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Важность признака')
            plt.title(f'Топ-{top_n} важных признаков: {model_name}')
            plt.tight_layout()
            plt.show()

    def compare_models(self):
        if not self.results:
            return

        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('R2', ascending=False)

        print("СРАВНЕНИЕ МОДЕЛЕЙ")

        for model_name in results_df.index:
            row = results_df.loc[model_name]
            print(f"{model_name:15} | R²: {row['R2']:.4f} | RMSE: {row['RMSE']:.4f} | Время: {row['Train_Time']:.2f}с")

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        results_df['R2'].sort_values().plot(kind='barh', color='skyblue')
        plt.title('Сравнение R² score')
        plt.xlabel('R²')

        plt.subplot(2, 2, 2)
        results_df['RMSE'].sort_values(ascending=False).plot(kind='barh', color='lightcoral')
        plt.title('Сравнение RMSE')
        plt.xlabel('RMSE')

        plt.subplot(2, 2, 3)
        results_df['Train_Time'].sort_values().plot(kind='barh', color='lightgreen')
        plt.title('Время обучения моделей')
        plt.xlabel('Время (секунды)')

        plt.subplot(2, 2, 4)
        results_df['CV_R2_Mean'].sort_values().plot(kind='barh', color='gold')
        plt.title('R² кросс-валидации')
        plt.xlabel('R²')

        plt.tight_layout()
        plt.show()

        return results_df


def main():
    analyzer = RegressionAnalyzer()

    print("Загрузка данных CalCOFI")
    data = analyzer.load_data()

    if data is None:
        return

    print("\nПодготовка данных")
    X_train, X_test, y_train, y_test = analyzer.prepare_features(data)

    if X_train is None:
        return

    print("\nОбучение и оценка моделей:")

    for model_name, model in analyzer.models.items():
        metrics, y_pred = analyzer.evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )

        if metrics is not None and y_pred is not None:
            analyzer.plot_predictions(y_test.values, y_pred, model_name)

            if hasattr(model, 'feature_importances_'):
                analyzer.plot_feature_importance(
                    model, X_train.columns.tolist(), model_name
                )

    results_df = analyzer.compare_models()

    if results_df is not None:
        best_model_name = results_df.index[0]
        best_model_metrics = results_df.iloc[0]

        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
        print(f"R²: {best_model_metrics['R2']:.4f}")
        print(f"RMSE: {best_model_metrics['RMSE']:.4f}")
        print(f"Общее время обучения всех моделей: {results_df['Train_Time'].sum():.2f}с")


if __name__ == "__main__":
    main()
