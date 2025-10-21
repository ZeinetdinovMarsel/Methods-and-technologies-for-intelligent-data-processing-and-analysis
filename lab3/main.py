import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    bottle_data = pd.read_csv('bottle.csv',low_memory=False)
    cast_data = pd.read_csv('cast.csv',low_memory=False)

    print(f"Размер bottle dataset: {bottle_data.shape}")
    print(f"Размер cast dataset: {cast_data.shape}")

    merged_data = bottle_data.merge(cast_data, on='Cst_Cnt', how='left', suffixes=('_bottle', '_cast'))

    print(f"Размер объединенного dataset: {merged_data.shape}")

    return merged_data


def prepare_regression_data(data, target_column='T_degC'):

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    missing_threshold = 0.1
    missing_ratio = data[numeric_columns].isnull().mean()
    valid_columns = missing_ratio[missing_ratio < missing_threshold].index.tolist()

    if target_column not in valid_columns:
        print(f"Целевая переменная {target_column} отсутствует или имеет много пропусков")
        potential_targets = ['Salnty', 'O2ml_L', 'STheta', 'O2Sat']
        for pt in potential_targets:
            if pt in numeric_columns and missing_ratio[pt] < missing_threshold:
                target_column = pt
                print(f"Используем альтернативную целевую переменную: {target_column}")
                break

    if target_column not in valid_columns:
        print("Не удалось найти подходящую целевую переменную")
        return None, None, None, None

    if target_column in valid_columns:
        valid_columns.remove(target_column)

    features = valid_columns[:20]

    print(f"Целевая переменная: {target_column}")
    print(f"Количество признаков: {len(features)}")
    print(f"Признаки: {features}")

    X = data[features].copy()
    y = data[target_column].copy()
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    split_ratio = 0.7
    split_index = int(len(X) * split_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X_train, X_test, y_train, y_test, target_column


def analyze_regression_performance(y_true, y_pred, model_name):
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }

    print(f"\n{model_name} - Метрики:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{model_name}\nПредсказания vs Истинные значения')

    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('Остатки регрессии')

    plt.tight_layout()
    plt.show()

    return metrics


def perform_depth_analysis(data, models, target_column='T_degC'):
    if 'Depthm' not in data.columns:
        print("Столбец 'Depthm' не найден для анализа по глубине")
        return

    data['depth_category'] = pd.cut(data['Depthm'],
                                    bins=[0, 50, 200, 500, 1000, 5000],
                                    labels=['0-50m', '50-200m', '200-500m', '500-1000m', '1000m+'])

    depth_results = []

    for depth_cat in data['depth_category'].cat.categories:
        depth_data = data[data['depth_category'] == depth_cat]

        if len(depth_data) < 50:
            continue

        X_depth = depth_data.select_dtypes(include=[np.number]).drop(columns=[target_column, 'Depthm'], errors='ignore')
        y_depth = depth_data[target_column]

        imputer = SimpleImputer(strategy='median')
        X_depth = pd.DataFrame(imputer.fit_transform(X_depth), columns=X_depth.columns)

        scaler = StandardScaler()
        X_depth = pd.DataFrame(scaler.fit_transform(X_depth), columns=X_depth.columns)

        valid_indices = y_depth.notna()
        X_depth = X_depth[valid_indices]
        y_depth = y_depth[valid_indices]

        if len(X_depth) < 20:
            continue

        for model in models:
            model_name = model.__class__.__name__

            try:
                cv = KFold(n_splits=min(5, len(X_depth)), shuffle=True, random_state=42)
                cv_scores = cross_validate(model, X_depth, y_depth, cv=cv,
                                           scoring=['neg_mean_squared_error', 'r2'],
                                           n_jobs=-1)

                depth_results.append({
                    'depth_category': depth_cat,
                    'model': model_name,
                    'RMSE': np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean()),
                    'R2': cv_scores['test_r2'].mean(),
                    'samples': len(X_depth)
                })

            except Exception as e:
                print(f"Ошибка при анализе глубины {depth_cat} для модели {model_name}: {e}")

    if depth_results:
        depth_df = pd.DataFrame(depth_results)
        print("\nАнализ производительности по глубинам:")
        print(depth_df.pivot_table(index='depth_category', columns='model', values='R2'))

        plt.figure(figsize=(12, 6))
        for model in depth_df['model'].unique():
            model_data = depth_df[depth_df['model'] == model]
            plt.plot(model_data['depth_category'], model_data['R2'], marker='o', label=model)

        plt.xlabel('Категория глубины')
        plt.ylabel('R² Score')
        plt.title('Производительность моделей по глубинам')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    print("Загрузка данных CalCOFI...")
    data = load_and_preprocess_data()

    if data is None:
        return

    print("\nПодготовка данных для регрессии...")
    X_train, X_test, y_train, y_test, target_column = prepare_regression_data(data)

    if X_train is None:
        return

    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        KNeighborsRegressor(n_neighbors=5)
    ]

    results_summary = []

    print("\nОбучение и оценка моделей регрессии...")

    for model in models:
        model_name = model.__class__.__name__
        print(f"Модель: {model_name}")

        try:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_validate(model, X_train, y_train, cv=cv,
                                       scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
                                       n_jobs=-1, return_train_score=False)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = analyze_regression_performance(y_test, y_pred, model_name)

            results_summary.append({
                'model': model_name,
                'RMSE_CV': np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean()),
                'MAE_CV': -cv_scores['test_neg_mean_absolute_error'].mean(),
                'R2_CV': cv_scores['test_r2'].mean(),
                'RMSE_test': metrics['RMSE'],
                'MAE_test': metrics['MAE'],
                'R2_test': metrics['R2']
            })

            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_names = X_train.columns
                indices = np.argsort(importance)[-10:]

                plt.figure(figsize=(10, 6))
                plt.barh(range(len(indices)), importance[indices], align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.title(f"Важность признаков: {model_name}")
                plt.xlabel("Значимость")
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Ошибка при обучении модели {model_name}: {e}")
            continue

    if results_summary:
        results_df = pd.DataFrame(results_summary)
        print("ИТОГИ ИССЛЕДОВАНИЯ РЕГРЕССИИ")
        results_df = results_df.sort_values('R2_test', ascending=False)
        print(results_df.round(4))

        plt.figure(figsize=(12, 8))

        models_list = results_df['model'].tolist()
        r2_scores = results_df['R2_test'].tolist()
        rmse_scores = results_df['RMSE_test'].tolist()

        x_pos = np.arange(len(models_list))

        plt.subplot(2, 1, 1)
        bars = plt.bar(x_pos, r2_scores, alpha=0.7, color='skyblue')
        plt.ylabel('R² Score (тест)')
        plt.title('Сравнение моделей регрессии: R² Score')
        plt.xticks(x_pos, models_list, rotation=45)

        for bar, value in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.subplot(2, 1, 2)
        bars = plt.bar(x_pos, rmse_scores, alpha=0.7, color='lightcoral')
        plt.ylabel('RMSE (тест)')
        plt.title('Сравнение моделей регрессии: RMSE')
        plt.xticks(x_pos, models_list, rotation=45)

        for bar, value in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    print("\nПроведение анализа по глубинам...")
    perform_depth_analysis(data, models, target_column)


if __name__ == "__main__":
    main()
