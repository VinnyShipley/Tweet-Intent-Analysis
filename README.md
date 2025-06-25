# Tweet Intent Classification Project

This project explores predicting customer support intent from tweets using machine learning. The goal is to build a model that accurately classifies tweet intents such as billing issues, complaints, or praise, helping improve automated customer support systems.

## Dataset

The analysis uses a manually labeled "golden set" of tweets, containing around 300 examples with intent labels. This subset was chosen to ensure clean, reliable evaluation, even though the full dataset contains many more unlabeled tweets.

## Approach

I started with a baseline text representation using TF-IDF vectorization, which captures the important words and phrases in each tweet. Then, I experimented with adding extra features like:

- **Tweet length** (number of characters)  
- **Is the tweet a question?** (binary feature)  
- **Sentiment score** (numeric sentiment polarity)  

Models were trained using logistic regression, and performance was evaluated using macro F1 scores to equally weigh all intent categories.

## Results

The macro F1 score improved slightly when adding sentiment as a feature, indicating that the emotional tone of tweets helps the model understand customer intent better. Adding tweet length or question detection did not significantly improve results.

Breaking down performance per intent revealed that sentiment helped most with detecting complaints — which makes sense given the emotional nature of negative feedback.

## Conclusion & Next Steps

Sentiment is a promising additional feature for tweet intent classification, especially for emotionally charged intents like complaints. However, more data and feature experimentation is needed to boost overall accuracy.

Future work could include combining multiple features and expanding the labeled dataset. Another next step could be experimenting with contextual embeddings like BERT could significantly improve performance, especially for intents where tone and nuance matter. TF-IDF is a strong baseline, but BERT would allow the model to better understand semantic meaning.

## Author
Vincent Shipley