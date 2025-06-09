import pandas as pd
import re
import emoji


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

# Assign intent function
def assign_intent(text, keywords):
    if not isinstance(text, str):
        return "Other"
    
    for intent, kws in keywords.items():
        for kw in kws:
            if kw in text:
                return intent
    return "Other"