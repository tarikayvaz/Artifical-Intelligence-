import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Veri setini yükleme (encoding='ISO-8859-1' ile)
data = pd.read_csv('dl_dataset.csv', header=None, names=['sentiment', 'id', 'date', 'query', 'user', 'text'], encoding='ISO-8859-1')

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
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression modeli eğitme
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Modeli ve vectorizer'ı kaydetme
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
