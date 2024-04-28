from flask import Flask, render_template, request, flash
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
import csv
import soundfile as sf
import numpy as np

app = Flask(__name__)
app.secret_key = 'KeySoUniquYouWontGuessIt'
# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("tiny", device=device)

# Load the BERT tokenizer and classifier
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

profanity_txt=open('profanityList.txt','r')
profanity_txt=profanity_txt.read().strip()
profanity_list=profanity_txt.split('\n')

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

    for profane_word in profanity_list:
        pattern = r'\b{}\b'.format(re.escape(profane_word))
        highlighted_text = re.sub(pattern, '<span style="color:red; font-style: italic;">{}</span>'.format(profane_word), highlighted_text, flags=re.IGNORECASE)

    return highlighted_text



@app.route('/', methods=['GET', 'POST'])
def text():
    if request.method == 'POST':
        user_input = request.form['user_input']
        #driver
        # return render_template('text.html', sentenceCheck=highlight(user_input), prediction_result='hallehuaeja', class_probabilities={1:1,2:2})
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
            prediction_result = f" {class_id_to_label[torch.argmax(logits).item()]}"
            return render_template('text.html', sentenceCheck=highlight(user_input), prediction_result=prediction_result, class_probabilities=class_probabilities)
        elif 'feedback' in request.form:
            suggested_class=request.form['feedback']
            user_input=user_input.replace('<span style="color:red; font-style: italic;">','')
            user_input=user_input.replace('</span>','')
            with open('suggestions.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_input, suggested_class])
            flash('Your suggestion has been recorded successfully!', 'success')
            return render_template('text.html')
            
    return render_template('text.html')

@app.route('/video', methods=['GET', 'POST'])
def vid():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_input = video_transcription(user_input)
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
            prediction_result = f" {class_id_to_label[torch.argmax(logits).item()]}"
            return render_template('vid.html', sentenceCheck=highlight(user_input), prediction_result=prediction_result, class_probabilities=class_probabilities)
        elif 'feedback' in request.form:
            suggested_class=request.form['feedback']
            user_input=user_input.replace('<span style="color:red; font-style: italic;">','')
            user_input=user_input.replace('</span>','')
            with open('suggestions.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_input, suggested_class])
            flash('Your suggestion has been recorded successfully!', 'success')
            return render_template('vid.html')
            

    return render_template('vid.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        user_input = request.files['user_input']
        user_input = audio_transcription(user_input)
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
            prediction_result = f" {class_id_to_label[torch.argmax(logits).item()]}"
            return render_template('audio.html', sentenceCheck=highlight(user_input), prediction_result=prediction_result, class_probabilities=class_probabilities)
        elif 'feedback' in request.form:
            suggested_class=request.form['feedback']
            user_input=user_input.replace('<span style="color:red; font-style: italic;">','')
            user_input=user_input.replace('</span>','')
            with open('suggestions.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_input, suggested_class])
            flash('Your suggestion has been recorded successfully!', 'success')
            return render_template('audio.html')
            

    return render_template('audio.html')





def video_transcription(video_URL):
    video = YouTube(video_URL)
    audio = video.streams.filter(only_audio=True).first()
    audio_file = audio.download()

    result = whisper_model.transcribe(audio_file)

    # Delete audio file after transcription
    os.remove(audio_file)

    return result["text"]

def audio_transcription(audio_file):
    with audio_file as f:
        audio_data, sample_rate = sf.read(f)

    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
    
    result = whisper_model.transcribe(audio_data, sample_rate)
    # Delete audio file after transcription
    # os.remove(audio_file)
    return result["text"]

if __name__ == '__main__':
    app.run(debug=True)
