## Cyberbullying Detection and Classification

## Research Published Paper Link :https://sciendo.com/article/10.2478/jsiot-2023-0020

# PreProcessing:
## 1)Tokenization:
This initial step breaks down each tweet into individual words and punctuation marks.
It helps the model understand the structure of the text and analyze word relationships.
Tokenization allows the model to focus on meaningful units of information for classification.

## 2)Stemming and Lemmatization:
Both techniques aim to reduce words to their base form, but with different approaches.
Stemming: Removes suffixes regardless of grammatical rules (e.g., "running" becomes "run").
Lemmatization: Identifies the dictionary form of a word based on its grammatical context (e.g.,
"running" becomes "run").

## 3)Stop Word Removal

This technique removes common words like "the," "a," and "is" that don't carry significant
meaning.
By removing stop words, the model can focus on words that contribute more to the sentiment and
context of the tweet.
This can improve the model's accuracy in identifying cyberbullying, as these words rarely
contribute to the offensive or harmful nature of the content.

## 4)K-Fold Cross Validation Technique:
Robust Model Evaluation: Applying k-fold cross-validation ensures that the model's performance
is rigorously evaluated across multiple folds of data. This helps in assessing the model's
generalization ability and reduces overfitting.

# Model:
## 1)Bert Model: 
Utilize the BERT (Bidirectional Encoder Representations from Transformers) model for textual
analysis and classification.
Fine-tune the BERT model using labeled data to classify the preprocessed audio transcripts into
cyberbullying and non-cyberbullying categories.

# Frontend:

StartUp:

![alt text](<Screenshot 2024-08-18 at 2.09.27 PM.png>)

Text Prediction:

![alt text](<Screenshot 2024-08-18 at 2.09.35 PM.png>)

Video Prediction:

![alt text](<Screenshot 2024-08-18 at 2.09.45 PM.png>)

Audio Prediction:

![alt text](<Screenshot 2024-08-18 at 2.09.54 PM.png>)