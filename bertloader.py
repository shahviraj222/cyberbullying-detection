# bertloder.py
# Install necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Text cleaning
import re
import string
import emoji
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set seed for reproducibility
import random
seed_value = 2042
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Set style for plots
sns.set_style("whitegrid")
sns.despine()
# plt.style.use("seaborn-whitegrid")
plt.style.use("default")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)

# Define stop words for text cleaning
stop_words = set(stopwords.words('english'))
batch_size = 32
# Initialize lemmatizer for text cleaning
lemmatizer = WordNetLemmatizer()

# Load the profanity list
profanity_txt = open('/Users/virajshah/Desktop/cyberbullying/profanityList.txt', 'r')
profanity_txt = profanity_txt.read().strip()
profanity_list = profanity_txt.split('\n')

def highlight(text):
    highlighted_text = text

    for profane_word in profanity_list:
        pattern = r'\b{}\b'.format(re.escape(profane_word))
        highlighted_text = re.sub(pattern, '\033[91m{}\033[0m'.format(profane_word), highlighted_text, flags=re.IGNORECASE)

    return highlighted_text

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def bert_tokenizer(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# Define the BERT classifier model
class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        n_input = 768
        n_hidden = 50
        n_output = 6

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# Instantiate the BERT classifier model
model = Bert_Classifier(freeze_bert=False)

# Load the trained model weights
model_path = '/Users/virajshah/Desktop/cyberbullying/bertmodel.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

# Mapping of class IDs to labels
class_id_to_label = {0: 'age', 1: 'ethnicity', 2: 'gender', 3: 'not_cyberbullying', 4: 'other_cyberbullying', 5: 'religion'}

# Input sentence for prediction
sentence = """Pat Dixon on X: "Just want to say what a fckn blast it’s been, working some cool venues, driving around this beautiful country and touring w @aFeliciaWorld. It’s NOTHING like being a gay rape victim who’s obsessed w his rapist, or even a pathetic drunk. So grateful for that. https://t.co/IhcahTuMv8" / X (twitter.com)"""

# Tokenize the input sentence
test_inputs, test_masks = bert_tokenizer([sentence])

# Apply the model for prediction
model.eval()
logits = model(test_inputs, test_masks)

# Convert logits to probabilities and predict the class
probs = nn.functional.softmax(logits, dim=-1)
predicted_class = torch.argmax(probs, dim=-1)
predicted_label = class_id_to_label[predicted_class.item()]

# Display the highlighted sentence, predicted class, and class probabilities
sentenceCheck = highlight(sentence)
print(sentenceCheck)
print()
print("Predicted class:", predicted_label)

probabilitys = probs.tolist()[0]
for idx, prob in enumerate(probabilitys):
    print(class_id_to_label[idx], " : ", prob * 100)
