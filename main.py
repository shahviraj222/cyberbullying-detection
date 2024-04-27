from flask import Flask, render_template, request
import torch
from pytube import YouTube
import whisper
import os
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch.nn as nn
from transformers import BertModel, BertTokenizer

app = Flask(__name__)

# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("tiny", device=device)

# Load the BERT tokenizer and classifier
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class Bert_Classifier(nn.Module):
    def __init__(self):
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

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# Instantiate the BERT classifier model
model = Bert_Classifier()

# Load the trained model weights
model_path = '/Users/virajshah/Desktop/cyberbullying/bertmodel.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

# Mapping of class IDs to labels
class_id_to_label = {0: 'age', 1: 'ethnicity', 2: 'gender', 3: 'not_cyberbullying', 4: 'other_cyberbullying', 5: 'religion'}

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

def highlight(text):
    highlighted_text = text
    # Your existing highlighting code
    return highlighted_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        if 'predict' in request.form:
            # Tokenize the user input using bert_tokenizer
            test_inputs, test_masks = bert_tokenizer([user_input])

            # Apply the model for prediction
            model.eval()
            logits = model(test_inputs, test_masks)

            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1)[0].tolist()

            # Format probabilities for display
            class_probabilities = {class_id_to_label[i]: round(prob * 100,2) for i, prob in enumerate(probabilities)}

            # Display the highlighted sentence, predicted class, and class probabilities
            sentenceCheck = highlight(user_input)
            prediction_result = f"Predicted class: {class_id_to_label[torch.argmax(logits).item()]}"
            return render_template('index.html', sentenceCheck=sentenceCheck, prediction_result=prediction_result, class_probabilities=class_probabilities)
        
        elif 'transcribe' in request.form:
            user_input = request.form['user_input']
            transcribed_text = video_transcription(user_input)
            return render_template('index.html', transcribed_text=transcribed_text)

    return render_template('index.html')

def video_transcription(video_URL):
    video = YouTube(video_URL)
    audio = video.streams.filter(only_audio=True).first()
    audio_file = audio.download()

    result = whisper_model.transcribe(audio_file)

    # Delete audio file after transcription
    os.remove(audio_file)

    return result["text"]

if __name__ == '__main__':
    app.run(debug=True)
