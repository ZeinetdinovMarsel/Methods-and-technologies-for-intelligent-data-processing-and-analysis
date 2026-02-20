import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import time
import seaborn as sns

warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    def __init__(self):
        self.results = {}
        self.models = {
            'KMeans': KMeans(n_clusters=2, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=2),
            'GaussianMixture': GaussianMixture(n_components=2, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'MeanShift': MeanShift()
        }

    def load_data(self):

        data = pd.read_csv('netflix_titles.csv')
        print(f"Загружены данные Netflix: {data.shape}")
        print("\nОбзор датасета:")
        print(f"Всего записей: {len(data)}")
        print("\nТипы столбцов:")
        print(data.dtypes)
        print("\nПропущенные значения по столбцам:")
        print(data.isnull().sum())
        return data

    def prepare_features(self, data):
        data = data.copy()

        print("\nПодготовка признаков для кластеризации...")

        if 'type' in data.columns:
            data['type_num'] = data['type'].map({'Movie': 1, 'TV Show': 0})
            print(f"\nРаспределение типов контента:")
            print(data['type'].value_counts(normalize=True))

        if 'rating' in data.columns:
            rating_mapping = {
                'TV-Y': 0, 'TV-Y7': 0, 'TV-G': 1, 'G': 1,
                'TV-PG': 2, 'PG': 2, 'PG-13': 3,
                'TV-14': 3, 'TV-MA': 4, 'R': 4, 'NC-17': 4,
                'NR': 2, 'UR': 2
            }
            most_common = data['rating'].mode()[0] if not data['rating'].empty else 'TV-MA'
            data['rating_num'] = data['rating'].fillna(most_common).map(rating_mapping).fillna(2)
            print(f"\nРаспределение рейтингов:")
            print(data['rating'].value_counts(normalize=True).head())

        if 'release_year' in data.columns:
            print(f"\nДиапазон годов выпуска: {data['release_year'].min()} - {data['release_year'].max()}")

        feature_cols = []
        if 'type_num' in data.columns:
            feature_cols.append('type_num')
        if 'release_year' in data.columns:
            feature_cols.append('release_year')
        if 'rating_num' in data.columns:
            feature_cols.append('rating_num')

        if not feature_cols:
            print("Подходящие признаки для кластеризации не найдены!")
            return None

        features = data[feature_cols].copy()
        print(f"\nИспользуемые признаки: {feature_cols}")

        for col in features.columns:
            if features[col].isnull().sum() > 0:
                features[col] = features[col].fillna(features[col].median())
                print(f"Пропущенные значения в {col} заполнены медианой: {features[col].median():.2f}")

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        print(f"\nСтатистика признаков после масштабирования:")
        for i, col in enumerate(feature_cols):
            print(
                f"{col}: среднее={scaled_features[:, i].mean():.4f}, стандартное отклонение={scaled_features[:, i].std():.4f}")

        return pd.DataFrame(scaled_features, columns=feature_cols)

    def evaluate_clustering(self, model, X, model_name):

        start_time = time.time()

        X_clean = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X_clean)
        else:
            model.fit(X_clean)
            labels = model.predict(X_clean)

        train_time = time.time() - start_time
        unique_labels = np.unique(labels)

        if -1 in unique_labels:
            n_clusters = len(unique_labels) - 1
            n_noise = np.sum(labels == -1)
            print(f"\n{model_name} нашел {n_clusters} кластеров и {n_noise} шумовых точек")
        else:
            n_clusters = len(unique_labels)
            print(f"\n{model_name} нашел {n_clusters} кластеров")

        print(f"Распределение кластеров:")
        for label in sorted(unique_labels):
            count = np.sum(labels == label)
            percentage = count / len(labels) * 100
            if label == -1:
                print(f"  Шумовые точки: {count} ({percentage:.1f}%)")
            else:
                print(f"  Кластер {label}: {count} ({percentage:.1f}%)")

        silhouette = silhouette_score(X_clean, labels) if n_clusters > 1 else -1
        calinski = calinski_harabasz_score(X_clean, labels) if n_clusters > 1 else -1
        davies = davies_bouldin_score(X_clean, labels) if n_clusters > 1 else -1

        metrics = {
            'n_clusters': n_clusters,
            'Silhouette': silhouette,
            'Calinski_Harabasz': calinski,
            'Davies_Bouldin': davies,
            'Train_Time': train_time
        }
        self.results[model_name] = metrics

        print(f"Метрики {model_name}:")
        print(f"  Индекс силуэта: {silhouette:.4f}")
        print(f"  Индекс Калинского-Харабаша: {calinski:.4f}")
        print(f"  Индекс Дэвиса-Болдуина: {davies:.4f}")
        print(f"  Время обучения: {train_time:.2f} секунд")

        return labels


    def plot_clusters_2d(self, X, labels, model_name):
        try:
            X_clean = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
            if X_clean.shape[1] > 1:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_clean)
                explained_var = pca.explained_variance_ratio_
                print(f"\nДоля объясненной дисперсии PCA: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")
            else:
                X_pca = np.column_stack((X_clean, np.zeros(X_clean.shape[0])))
                explained_var = [1.0, 0.0]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', s=50, alpha=0.8, edgecolors='w',
                                  linewidth=0.5)
            plt.colorbar(scatter, label='ID кластера')
            plt.title(f'Кластеры {model_name} (проекция PCA)', fontsize=14, fontweight='bold')
            plt.xlabel(f'Главная компонента 1 ({explained_var[0]:.1%} дисперсии)', fontsize=12)
            plt.ylabel(f'Главная компонента 2 ({explained_var[1]:.1%} дисперсии)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Ошибка при построении графика {model_name}: {str(e)}")


    def compare_models(self):
        results_df = pd.DataFrame(self.results).T

        print("СРАВНЕНИЕ МОДЕЛЕЙ КЛАСТЕРИЗАЦИИ")

        results_df = results_df.sort_values('Silhouette', ascending=False)

        for idx, row in results_df.iterrows():
            print(f"\n{idx}:")
            print(f"  Кластеров: {row['n_clusters']}")
            print(f"  Индекс силуэта: {row['Silhouette']:.4f}")
            print(f"  Индекс Калинского-Харабаша: {row['Calinski_Harabasz']:.4f}")
            print(f"  Индекс Дэвиса-Болдуина: {row['Davies_Bouldin']:.4f}")
            print(f"  Время обучения: {row['Train_Time']:.2f} секунд")

        if len(results_df) > 1:
            self._create_comparison_plots(results_df)

        return results_df


    def _create_comparison_plots(self, results_df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Сравнение алгоритмов кластеризации', fontsize=16, fontweight='bold')

        metrics = [
            ('Silhouette', 'Индекс силуэта', 'skyblue'),
            ('Calinski_Harabasz', 'Индекс Калинского-Харабаша', 'lightgreen'),
            ('Davies_Bouldin', 'Индекс Дэвиса-Болдуина', 'salmon'),
            ('Train_Time', 'Время обучения', 'gold')
        ]

        for i, (metric, title, color) in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            values = results_df[metric].copy()
            if metric == 'Davies_Bouldin':
                values = values.replace(-1, np.nan)

            if not values.isna().all():
                values.plot(kind='barh', ax=ax, color=color, edgecolor='black')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Значение', fontsize=10)
                ax.grid(axis='x', alpha=0.3)

                for j, v in enumerate(values):
                    if not pd.isna(v):
                        ax.text(v + 0.01 * (values.max() - values.min()), j, f'{v:.2f}',
                                va='center', fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def main():
    analyzer = ClusteringAnalyzer()
    data = analyzer.load_data()

    X = analyzer.prepare_features(data)

    print("ЗАПУСК АЛГОРИТМОВ КЛАСТЕРИЗАЦИИ")

    for model_name, model in analyzer.models.items():
        print(f"Обработка: {model_name}")

        labels = analyzer.evaluate_clustering(model, X, model_name)
        if labels is not None:
            analyzer.plot_clusters_2d(X, labels, model_name)

    print("ИТОГОВЫЙ АНАЛИЗ")

    results_df = analyzer.compare_models()
    if results_df is not None and not results_df.empty:
        best_model = results_df.index[0]
        best_score = results_df.loc[best_model, 'Silhouette']

        print(f"\nЛучшая модель: {best_model}")
        print(f"Лучший индекс силуэта: {best_score:.4f}")


if __name__ == "__main__":
    main()
