import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Veri setini yükleme (encoding='ISO-8859-1' ile)
data = pd.read_csv('sampled_dataset.csv', header=None, names=['sentiment', 'id', 'date', 'query', 'user', 'text'], encoding='ISO-8859-1')

# Gereksiz sütunları kaldırma
data = data[['sentiment', 'text']]

# Metin temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # URL'leri kaldırma
    text = re.sub(r'@\w+', '', text)  # Kullanıcı adlarını kaldırma
    text = re.sub(r'#\w+', '', text)  # Hashtag'leri kaldırma
    text = re.sub(r'\d+', '', text)  # Sayıları kaldırma
    text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları kaldırma
    text = text.lower().strip()  # Küçük harfe çevirme ve boşlukları kaldırma
    return text

# Metinleri temizleme
data['text'] = data['text'].apply(clean_text)

# Eğitim ve test veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# TF-IDF vektörleştirme, max_features=1000
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Modellerin listesi
models = {
    "Support Vector Machine": SVC(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# Modellerin performansını değerlendirme
for model_name, model in models.items():
    print(f"Model: {model_name}")
    # Modeli eğitme
    model.fit(X_train_tfidf, y_train)
    # Modelin tahmin yapması
    y_pred = model.predict(X_test_tfidf)
    # Performans değerlendirmesi
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Cross-validation kullanarak modelin performansını değerlendirme
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)  # 5-fold cross-validation
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", np.mean(cv_scores))
    
    # Confusion matrix oluşturma ve görselleştirme
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    print("\n" + "-"*50 + "\n")
