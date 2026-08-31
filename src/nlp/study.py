import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
texts = [
    "I love this movie",
    "This movie is amazing",
    "What a fantastic film",
    "The acting was excellent",
    "I really enjoyed this",

    "I hate this movie",
    "This movie is terrible",
    "What a boring film",
    "The acting was awful",
    "I really disliked this",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",

    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
]


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

print("=" * 60)
print("VOCABULARY")
print("=" * 60)

print(vectorizer.get_feature_names_out())

print("\n" + "=" * 60)
print("VECTOR SHAPE")
print("=" * 60)

print(X.shape)

print("\nVí dụ vector của câu đầu tiên:")
print(X[0].toarray())

model = LogisticRegression()

model.fit(X, labels)

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    print("\nText:")
    print(f'  "{text}"')

    print("\nPrediction:")
    print(f"  {prediction}")

    print("\nConfidence:")
    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label:10}: {probability:.2%}")


predict_sentiment("I love this film")

predict_sentiment("This was the worst movie I have ever seen")

predict_sentiment("The movie was amazing but the ending was boring")



