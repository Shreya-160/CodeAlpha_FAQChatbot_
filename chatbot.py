from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is AI?",
    "What is Python?",
    "What is machine learning?",
    "Who developed Python?",
    "What is ChatGPT?"
]

answers = [
    "AI stands for Artificial Intelligence.",
    "Python is a programming language.",
    "Machine learning enables computers to learn from data.",
    "Python was developed by Guido van Rossum.",
    "ChatGPT is an AI chatbot developed by OpenAI."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

print("FAQ Chatbot Started")
print("Type 'exit' to quit")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    q = vectorizer.transform([query])

    similarity = cosine_similarity(q, X)

    index = similarity.argmax()

    print("Bot:", answers[index])