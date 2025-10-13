import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    train_data = pd.read_csv('train_gr/train.csv')
    test_data = pd.read_csv('test_gr/test.csv')
    game_overview = pd.read_csv('train_gr/game_overview.csv')

    train_merged = train_data.merge(game_overview, on='title', how='left')
    test_merged = test_data.merge(game_overview, on='title', how='left')
    return train_merged, test_merged

train_data, test_data = load_data()
X_train = train_data['user_review']
y_train = train_data['user_suggestion']
X_test = test_data['user_review']

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = [
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    KNeighborsClassifier(n_neighbors=2),
    DecisionTreeClassifier(max_depth=35, random_state=42, class_weight='balanced'),
    RandomForestClassifier(n_estimators=100, random_state=42),
    MultinomialNB(),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
]

# split_reviews_char_count = int(train_data['user_review'].apply(len).median())

split_reviews_char_count = int(train_data['user_review'].apply(len).quantile(0.5))



train_data['review_length'] = train_data['user_review'].apply(len)
short_reviews_idx = train_data[train_data['review_length'] < split_reviews_char_count].index
long_reviews_idx = train_data[train_data['review_length'] >= split_reviews_char_count].index

data_subsets = {
    f'Короткие обзоры (<{split_reviews_char_count} символов)': short_reviews_idx,
    f'Длинные обзоры (>={split_reviews_char_count} символов)': long_reviews_idx,
    'Все обзоры': train_data.index
}

def analyze_errors(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок: {model_name}')
    plt.xlabel('Предсказано')
    plt.ylabel('Истинное значение')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=['НЕ РЕКОМЕНДУЕТ', 'РЕКОМЕНДУЕТ']))

def predict_single_review(text, model, vectorizer):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    result = "РЕКОМЕНДУЕТ" if prediction == 1 else "НЕ РЕКОМЕНДУЕТ"
    confidence = probability[1] if prediction == 1 else probability[0]
    return result, confidence, probability

results_summary = []

for subset_name, subset_idx in data_subsets.items():
    X_subset = X_train_tfidf[subset_idx, :]
    y_subset = y_train.iloc[subset_idx]

    print(f"\nПодсет: {subset_name}")

    best_f1 = 0
    best_model_name = None

    min_class_count = y_subset.value_counts().min()
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        print(f"Пропускаем {subset_name} — слишком мало примеров для кросс-валидации")
        continue

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model in models:
        model_name = model.__class__.__name__

        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(model, X_subset, y_subset, cv=cv, scoring=scoring)

        model.fit(X_subset, y_subset)
        y_pred = model.predict(X_subset)
        y_proba = model.predict_proba(X_subset)[:, 1]

        f1_mean = cv_results['test_f1'].mean()
        results_summary.append({
            'subset': subset_name,
            'model': model_name,
            'accuracy': cv_results['test_accuracy'].mean(),
            'precision': cv_results['test_precision'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'f1': f1_mean,
            'roc_auc': cv_results['test_roc_auc'].mean()
        })

        print(f"\nМодель: {model_name}")
        print(f"Средняя точность (Accuracy): {cv_results['test_accuracy'].mean():.4f}")
        print(f"Средняя точность по положительным классам (Precision): {cv_results['test_precision'].mean():.4f}")
        print(f"Средняя полнота (Recall): {cv_results['test_recall'].mean():.4f}")
        print(f"Среднее значение F1-метрики: {f1_mean:.4f}")
        print(f"Средняя площадь под ROC-кривой (ROC-AUC): {cv_results['test_roc_auc'].mean():.4f}")

        analyze_errors(y_subset, y_pred, model_name)

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = vectorizer.get_feature_names_out()
            indices = np.argsort(importance)[-20:]
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.title(f"Важность признаков: {model_name}")
            plt.xlabel("Значимость")
            plt.ylabel("Признак")
            plt.show()

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_model_name = model_name

    if best_model_name:
        print(f"\nРекомендуемая модель для '{subset_name}': {best_model_name} (F1: {best_f1:.4f})")

results_df = pd.DataFrame(results_summary)
print("\nИтоги:")
print(results_df.sort_values(['subset', 'f1'], ascending=[True, False]))
