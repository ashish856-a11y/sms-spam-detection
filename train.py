# Data & Libraries
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Download NLTK data
# nltk.download('punkt')
nltk.download('punkt', download_dir=nltk.data.find('tokenizers').path)

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# def transform_text(text):
#     text = text.lower()
#     tokens = nltk.word_tokenize(text)
#     tokens = [word for word in tokens if word.isalnum()]
#     tokens = [word for word in tokens if word not in stop_words]
#     tokens = [ps.stem(word) for word in tokens]
#     return " ".join(tokens)


from nltk.tokenize import RegexpTokenizer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

def transform_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)  # Safe tokenizer, avoids punkt issues
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)


df['transformed_text'] = df['text'].apply(transform_text)
df['num_characters'] = df['text'].apply(len)

# Split data on transformed_text
X = df[['transformed_text','num_characters']]
y = df['target']
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=3000), 'transformed_text'),
    ('num', MinMaxScaler(), ['num_characters'])  # ✅ List instead of string
])



pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MultinomialNB())
])



# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, 'spam_classifier_pipeline.pkl')

print("✅ Model pipeline trained and saved successfully.")
