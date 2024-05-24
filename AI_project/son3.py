import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Veri setini yükleme (encoding='ISO-8859-1' ile)
data = pd.read_csv('dl_dataset.csv', encoding='ISO-8859-1')

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

# Sınıfları sayısal değerlere dönüştürme
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# Eğitim ve test veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Tokenizasyon
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# One-hot encoding
num_classes = 2  # İki sınıf (pozitif ve negatif)
print(f"Number of classes: {num_classes}")

# One-hot encoding yapma
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Model oluşturma
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=32))
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modeli oluşturma ve eğitme
model = create_model()
history = model.fit(X_train_pad, y_train_cat, epochs=3, batch_size=32, validation_data=(X_test_pad, y_test_cat), verbose=2)

# Model performansını değerlendirme
loss, accuracy = model.evaluate(X_test_pad, y_test_cat, verbose=0)
print(f'Accuracy: {accuracy}')

# Tahminler yapma
y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test_cat, axis=1)

# Confusion matrix ve classification report
conf_matrix = confusion_matrix(y_test_labels, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test_labels, y_pred))

# Confusion matrix görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation yapma
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Daha az katlama
cv_accuracy = []

for train_index, val_index in kf.split(X_train_pad):
    X_train_kf, X_val_kf = X_train_pad[train_index], X_train_pad[val_index]
    y_train_kf, y_val_kf = y_train_cat[train_index], y_train_cat[val_index]

    model = create_model()
    model.fit(X_train_kf, y_train_kf, epochs=3, batch_size=32, verbose=1)  # Epochs sayısını ve batch size'ı azalttım
    loss, accuracy = model.evaluate(X_val_kf, y_val_kf, verbose=0)
    cv_accuracy.append(accuracy)

print("Cross-validation scores:", cv_accuracy)
print("Mean CV score:", np.mean(cv_accuracy))
