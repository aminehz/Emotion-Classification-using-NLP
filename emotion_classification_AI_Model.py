

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


import pandas as pd
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import numpy as np
import emoji
import re
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

"""# **Load Dataset**"""

data1 = pd.read_csv('emotion-labels-train.csv')

data1.head()

data2 = pd.read_csv('emotion-labels-test.csv')
data2.head()

data3 = pd.read_csv('emotion-labels-val.csv')
data3.head()

data = pd.concat([data1,data2,data3], ignore_index=True)
data.head()

data['label'].value_counts()

"""# **Text Cleaning**"""

def remove_punctuation(text):
  punctuationfree = "".join([i for i in text if i not in string.punctuation])
  return punctuationfree

data['text'] = data['text'].apply(lambda x:remove_punctuation(x))
data.head()

data['text'] = data['text'].apply(lambda x: x.lower())
data.head()

"""# **Tokenization**"""

data['text'] = data['text'].apply(word_tokenize)
data.head()

"""# **StopWords**"""

stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
  output=[i for i in text if i not in stopwords]
  return output

data['text'] = data['text'].apply(lambda x:remove_stopwords(x))
data.head()

"""# **Lemmatization**"""

wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text

data['text'] = data['text'].apply(lambda x:lemmatizer(x))
data.head()

"""# **Vectorization TF-IDF**"""

def join_token(tokens):
  if isinstance(tokens, list):
    return ' '.join(tokens)
  return str(tokens)
data['text'] = data['text'].apply(join_token)
data.head()

vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(data['text']).toarray()
y = data['label']

"""# **Split the data into train and test**"""

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])
y = to_categorical(y)

x_train, x_test, y_train , y_test = train_test_split(x,y, test_size=0.25, random_state=42)

"""# **Model Building**"""

model = Sequential([
    Dense(512, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=-1)
y_test_classes = y_test.argmax(axis=-1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy:.2f}')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

"""# OWN TEST"""

new_texts = [
    "I am feeling great today!",
    "I am so frustrated with everything.",
    "What a wonderful and exciting day!",
    "I am so angry about the situation."
]

# Preprocess new texts
new_texts_processed = [ ' '.join(text.split()) for text in new_texts ]  # Ensure text is in the same format

# Vectorize new texts using the same TF-IDF Vectorizer
new_texts_vectorized = vectorizer.transform(new_texts_processed).toarray()

# Make predictions
predictions = model.predict(new_texts_vectorized)

# Convert predictions to class labels
predicted_classes = predictions.argmax(axis=-1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Print predictions
for text, label in zip(new_texts, predicted_labels):
    print(f'Text: "{text}" - Predicted Emotion: {label}')

