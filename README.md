FileName: withML.py
To Run this : py withML.py (in command prompt)

-> Emotional Analysis using ML techniques
1) TF-IDF (For Feature Extraction)
2) Naive Bayes (ML Model to predict emotion)
3) Classifier used is MultinomialNB()

FileName: withoutML.py
To Run this : py withoutML.py (in command prompt)

-> Emotional Analysis without using ML techniques(only NLP techniques)
1) Oversampling of data (to give sad emotion more priority)
2) POS Tagging to determine Adverbs,Adjectives (because Adjective hold most of the emotion and Adverb tells how intense is the emotion is)
3) Bi-grams to negate the emotions which contain 'not'
4) Create Dictionaries for 6 emotions (Happy,sad,anger,fear,surprise,love) like word embeddings along with their weight)


Remaining files are used in these 2 files
