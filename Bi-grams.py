import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import demoji
import nltk
import re

df = pd.read_csv('new_dataset.csv')

def remove_emojis(text):
    return demoji.replace(text, '')


df['sentence']=df['sentence'].apply(remove_emojis)

custom_stopwords = set()
with open('stopwords.txt', 'r', encoding='utf-8') as file:
    for line in file:
        custom_stopwords.add(line.strip())

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]
    return ' '.join(filtered_words)

df['sentence'] = df['sentence'].apply(remove_stopwords)
def custom_tokenizer(text):
    words = text.split()
    result = []
    for i in range(len(words) - 1):
        if words[i] == "not":
            result.append("not " + words[i + 1])
    return result

def normalize_text(text):
    text = text.lower()

    #Remove Special Characters.
    text = re.sub(r"[^\w\s]", " ", text)
    return text
    
df['sentence']=df['sentence'].apply(normalize_text)
text_data = df['sentence'].tolist()

ngram_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
X = ngram_vectorizer.fit_transform(text_data)

# Get feature names (custom bigrams)
bi_grams = ngram_vectorizer.get_feature_names_out()
bi_grams_df = pd.DataFrame(bi_grams, columns=['Bi-Grams'])
bi_grams_df.to_csv('bi_grams.csv', index=False)
# Display the generated n-grams
print("Bi-Grams with not :")
print(bi_grams)