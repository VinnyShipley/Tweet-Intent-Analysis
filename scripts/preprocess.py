import re
import emoji
from textblob import TextBlob


#################  CLEANING #######################

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hastags, keeps the word, loses the #
    text = re.sub(r'#', '', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Lowercase all text
    text = text.lower()

    return text 


###################### LABELING ###################

def assign_intent(text, keywords):
    if not isinstance(text, str):
        return "Other"
    
    for intent, kws in keywords.items():
        for kw in kws:
            if kw in text:
                return intent
    return "Other"


################ IS QUESTION #######################

def is_question(text):
    question_words = ['who', 'what', 'when', 'where', 'why']
    text = text.lower()
    return int(text.endswith('?') or any(qw in text for qw in question_words))


################## SENTIMENT SCORE ##########################

def get_sentiment_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity
