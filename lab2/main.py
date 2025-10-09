import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data():
    train_data = pd.read_csv('train_gr/train.csv')
    test_data = pd.read_csv('test_gr/test.csv')
    game_overview = pd.read_csv('train_gr/game_overview.csv')

    train_merged = train_data.merge(game_overview, on='title', how='left')
    test_merged = test_data.merge(game_overview, on='title', how='left')
    return train_merged, test_merged


print("Загрузка данных")
train_data, test_data = load_data()

X_train = train_data['user_review']
y_train = train_data['user_suggestion']
X_test = test_data['user_review']

print("Векторизация текста")
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
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, verbose=1),
    KNeighborsClassifier(n_neighbors=2),
    DecisionTreeClassifier(max_depth=35, random_state=42, class_weight='balanced'),
    RandomForestClassifier(n_estimators=100, random_state=42, verbose=1),
    MultinomialNB(),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42, verbose=1),
]


def analyze_errors(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def predict_single_review(text, model, vectorizer):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    result = "РЕКОМЕНДУЕТ" if prediction == 1 else "НЕ РЕКОМЕНДУЕТ"
    confidence = probability[1] if prediction == 1 else probability[0]
    return result, confidence, probability


for model in models:

    name = model.__class__.__name__
    print(f"\nМодель: {name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    model.fit(X_train_tfidf, y_train)
    y_pred_train = model.predict(X_train_tfidf)
    y_proba_train = model.predict_proba(X_train_tfidf)[:, 1]

    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Precision: {precision_score(y_train, y_pred_train):.4f}")
    print(f"Recall: {recall_score(y_train, y_pred_train):.4f}")
    print(f"F1-score: {f1_score(y_train, y_pred_train):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}")

    analyze_errors(y_train, y_pred_train, name)

    test_predictions = model.predict(X_test_tfidf)
    test_probabilities = model.predict_proba(X_test_tfidf)
    results = test_data[['review_id', 'title']].copy()
    results['predicted_recommendation'] = test_predictions
    results['probability_negative'] = test_probabilities[:, 0]
    results['probability_positive'] = test_probabilities[:, 1]

    print("\nПримеры предсказаний:")
    demo_examples = test_data.head(5)
    for i, row in demo_examples.iterrows():
        result, confidence, proba = predict_single_review(row['user_review'], model, vectorizer)
        print(f"\nПример {i + 1} (ID: {row['review_id']}, Игра: {row['title']}):")
        print(f"Текст: {row['user_review'][:150]}...")
        print(f"Предсказание: {result}")
        print(f"Уверенность: {confidence:.2%}")
        print(f"Вероятности: [Не рекомендует: {proba[0]:.2%}, Рекомендует: {proba[1]:.2%}]")
