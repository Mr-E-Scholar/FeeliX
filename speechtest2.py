import speech_recognition as sr
import time
import pygame
import wave
import sounddevice as sd
import simpleaudio as sa
import numpy as np
import assemblyai as aai
import tkinter as tk
import json
import re
import threading
import math
from PIL import Image, ImageTk

from collections import Counter
from scipy.io.wavfile import write


def play_video(path):
    pass

def play_mp3(filename):
    print("Felix has woken!")
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(.5)
        
def record_audio(filename, seconds=5, fs=44100, channels=2):
    print("Now recording...")
    recording = sd.rec(int(seconds*fs), samplerate=fs, channels=channels)
    sd.wait() 
    recording_int16 = np.int16(recording / np.max(np.abs(recording))*32767)
    write(filename, fs, recording_int16)
    print("saved")
    wake_detected=False

def play_audio(filename):
    print("playing...")
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    print("finished playback")
    
def transcribe_audio(filename):
    print("Transcribing audio...")
    recognizer=sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(text)
        except sr.RequestError as e:
            print(e)
        return text
#     aai.settings.api_key = "b6b980fdbf064e4097501412d6d33e4c"
#     transcriber = aai.Transcriber()
#     path = "./"+filename
#     transcript = transcriber.transcribe(path)
#     print(transcript.text)

def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
def preprocess(sentence):
    return re.findall(r'\b\w+\b', sentence.lower())
    
def check_emotion(sentence, emotion_keywords):
    print("Checking the library...")
    words = preprocess(sentence)
    print(words)
    for word in words:
        for emotion, keywords in emotion_keywords.items():
            if word in keywords:
#                 print(f"{emotion} from {word}")
                print(emotion)
                return emotion, word  
    return emotion, wordch

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])

def update_display(emotion, word):
    if emotion == 'positive':
#         current_feeling.config(bg='green')
#         current_label.config(bg='green')
        create_circle(current_feeling, 'green')
    elif emotion == 'neutral':
        create_circle(current_feeling, 'yellow')
#         current_feeling.config(bg='yellow')
#         current_label.config(bg='yellow')
    elif emotion == 'negative':
        create_circle(current_feeling, 'red')
#         current_feeling.config(bg='red')
#         current_label.config(bg='red')
    with open('myfeelings.json', 'r') as file:
        mydata = json.load(file)
    mydata["feelings"].append(emotion)
    feels = mydata["feelings"]
    print(f"You said you are feeling {emotion} by using the keyword {word}")
    feelings_count = Counter(feels)
    pos_count = feelings_count["positive"]
    neut_count = feelings_count["neutral"]
    neg_count = feelings_count["negative"]
    total_count = pos_count + neut_count + neg_count
    max_feel = max(feelings_count, key=feelings_count.get)
    green_overall = math.floor((255/total_count * (pos_count + neut_count)))
    red_overall = math.floor((255/total_count * (neg_count + neut_count)))
    overall_color = (red_overall, green_overall, 0)
    
    create_circle(overall_feeling, rgb_to_hex(overall_color))
    
#     overall_feeling.config(bg=rgb_to_hex(overall_color))
#     overall_label.config(bg=rgb_to_hex(overall_color))


    if max_feel == "positive":
        create_circle(common_feeling, 'green')
#         common_feeling.config(bg='green')
#         common_label.config(bg='green')
    elif max_feel == "neutral":
        create_circle(common_feeling, 'yellow')
#         common_feeling.config(bg='yellow')
#         common_label.config(bg='yellow')
    elif max_feel == "negative":
        create_circle(common_feeling, 'red')
#         common_feeling.config(bg='red')
#         common_label.config(bg='red')
    print("Positives: ", pos_count)
    print("Neutrals: ", neut_count)
    print("Negatives: ", neg_count)
    print(f"Max count is: {feelings_count[max_feel]} for the emotion: {max_feel}")
    with open('myfeelings.json', 'w') as file:
        json.dump(mydata, file, indent=4)

def main(wake_word, device_index):
    emotion_keywords = load_data("library.json")
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index = device_index)
    first_time = True
    while True:
        wake_detected = False
        with microphone as source:
            if first_time:
                print("Beginning the app...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                if first_time:
                    print("Listening for the key word!")
                    first_time = False  
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=2)
                text = recognizer.recognize_google(audio)
#                 print(f"Recognized: {text}")
                if wake_word.lower() in text.lower():
                    print(f"Wake word '{wake_word}' detected!")
                    wake_detected=True
            except sr.UnknownValueError:
                print("Could not understand audio")     
            except sr.RequestError:
                print("Could not request results, check internet")         
            except sr.WaitTimeoutError:
                print("Listening timed out...")
                
        if wake_detected:
            mp3file = "voice.mp3"
            vidfile = "wakevidsmall.mov"
#             play_video(vidfile)
            play_mp3(mp3file)
            filename = "test1.wav"
            record_audio(filename)
            transcription = transcribe_audio(filename)
            emotion, word = check_emotion(transcription, emotion_keywords)
            update_display(emotion, word)


def main_thread(wake_word, device_index):
    threading.Thread(target=main, args=(wake_word, device_index)).start()
    
def create_circle(frame, color):
    radius=30
    canvas = tk.Canvas(frame, height=2*radius, width=2*radius, bg='black', highlightthickness=0)
    canvas.pack(side='left')
    
    x,y = radius,radius
    canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color)
#     canvas.create_oval(radius, radius, -radius, diameter-radius, fill=color)
    
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("400x300")



current_feeling = tk.Frame(root, height=100, width=200, bg='black')
overall_feeling = tk.Frame(root, height=100, width=200, bg='black')
common_feeling = tk.Frame(root, height=100, width=200, bg='black')
picFrame = tk.Frame(root, height=300, width=200)

current_feeling.pack(side='top', fill=None, anchor='w')
overall_feeling.pack(side='top', fill=None, anchor='w')
common_feeling.pack(side='top', fill=None, anchor='w')
picFrame.pack(side='top', fill=None, expand=True)

current_feeling.pack_propagate(False)
overall_feeling.pack_propagate(False)
common_feeling.pack_propagate(False)
picFrame.pack_propagate(False)

current_label = tk.Label(current_feeling, text="Current", bg='black', fg='white')
overall_label = tk.Label(overall_feeling, text="Overall", bg='black', fg='white')
common_label = tk.Label(common_feeling, text="Most Common", bg='black', fg='white')

current_label.pack(expand=True)
overall_label.pack(expand=True)
common_label.pack(expand=True)

# create_circle(current_feeling, 'green')
# create_circle(overall_feeling, 'yellow')
# create_circle(common_feeling, 'blue')

felimage = Image.open("felix.png")
photo = ImageTk.PhotoImage(felimage)
picLabel = tk.Label(picFrame, image=photo)
picLabel.image = photo
picLabel.pack()


                    
if __name__ == "__main__":
    wake_word = "felix"
    device_index = 3
    main_thread(wake_word, device_index)
    root.mainloop()