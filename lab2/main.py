import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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

print("Обучение модели")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)

cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')

print(f"Средняя точность: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("Предсказание на тестовых данных")
test_predictions = model.predict(X_test_tfidf)
test_probabilities = model.predict_proba(X_test_tfidf)

results = test_data[['review_id', 'title']].copy()
results['predicted_recommendation'] = test_predictions
results['probability_negative'] = test_probabilities[:, 0]
results['probability_positive'] = test_probabilities[:, 1]

if hasattr(model, 'coef_'):
    print(f"\nАнализ важных признаков для {type(model).__name__}")
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coefficients
    })

    top_positive = feature_importance.nlargest(10, 'importance')
    print("\nТоп-10 слов, указывающих на РЕКОМЕНДАЦИЮ:")
    for _, row in top_positive.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    top_negative = feature_importance.nsmallest(10, 'importance')
    print("\nТоп-10 слов, указывающих на НЕРЕКОМЕНДАЦИЮ:")
    for _, row in top_negative.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

def predict_single_review(text, model, vectorizer):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    result = "РЕКОМЕНДУЕТ" if prediction == 1 else "НЕ РЕКОМЕНДУЕТ"
    confidence = probability[1] if prediction == 1 else probability[0]

    return result, confidence, probability

print("\nРабота модели")

demo_examples = test_data.head(5)

for i, row in demo_examples.iterrows():
    result, confidence, proba = predict_single_review(row['user_review'], model, vectorizer)

    print(f"\nПример {i + 1} (ID: {row['review_id']}, Игра: {row['title']}):")
    print(f"Текст: {row['user_review'][:150]}...")
    print(f"Предсказание: {result}")
    print(f"Уверенность: {confidence:.2%}")
    print(f"Вероятности: [Не рекомендует: {proba[0]:.2%}, Рекомендует: {proba[1]:.2%}]")