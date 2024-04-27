# -*- coding: utf-8 -*-
"""videoTranscript.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wn9Gyx90jONYxImsg99IOLSIDxQJ4_Tb
"""

# Install necessary dependencies
# !pip install pytube pydub SpeechRecognition
# !apt-get install -y graphviz && pip install pydot

# Install whisper library
# !pip install git+https://github.com/openai/whisper.git

# path where model is saved ls /Users/virajshah/.cache/whisper/

import torch
import os
from pytube import YouTube
import pydot
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = whisper.load_model("tiny", device=device)

def video_to_audio(video_URL, destination, final_filename):
    video = YouTube(video_URL)
    audio = video.streams.filter(only_audio=True).first()

    output = audio.download(output_path=destination)

    _, ext = os.path.splitext(output)
    new_file = final_filename + '.mp3'

    os.rename(output, new_file)

# Video to audio
video_URL = 'https://youtu.be/VSVjWnM1Wmw?si=2BqnCM3VesrDYGRs'
destination = "."
final_filename = "motivational_speech"
video_to_audio(video_URL, destination, final_filename)

# Audio to text
audio_file = "motivational_speech.mp3"
result = whisper_model.transcribe(audio_file)
print(result["text"])
