import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load dataset (change filename if needed)
df = pd.read_csv('sms_spam.csv', encoding='latin-1', usecols=[0,1], names=['label','message'], header=0)

# Split into train and test (80% train, 20% test)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Create pipeline
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Train model
model.fit(train['message'], train['label'])

# Prediction function
def predict_message(text):
    pred_prob = model.predict_proba([text])[0]  # [prob_ham, prob_spam]
    pred_class = model.predict([text])[0]
    # Return probability of spam (index 1) and predicted label
    return [round(pred_prob[1], 2), pred_class]
    

# Test example messages
if __name__ == "__main__":
    print(predict_message("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"))
    print(predict_message("Hey, are we still meeting for lunch today?"))
# ... all your imports and model training code above ...

# Updated predict_message function
def predict_message(text):
    pred_prob = model.predict_proba([text])[0]
    pred_class = model.predict([text])[0]
    return [round(float(pred_prob[1]), 2), str(pred_class)]

