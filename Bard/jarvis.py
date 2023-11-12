import pyttsx3
import speech_recognition as sr
from bard_clone import Bard_Clone as bc

engine = pyttsx3.init()

def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print('Skipping unknown error')

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def loop():
    while True:
        print("Say Jarvis to start")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio)
            if transcription.lower() == "jarvis":
                filename = "input.wav"
                response = bc('Hello')
                complresp = response + ' Whats your question?'
                speak_text(response)
                with sr.Microphone() as source:
                    recognizer = sr.Recognizer()
                    source.pause_threshold = 1
                    audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                    with open(filename, 'wb') as f:
                        f.write(audio.get_wav_data())
                text = transcribe_audio_to_text(filename)
                if text:
                    print(f'You said: {text}')
                    response = bc(text)
                    print(f"Jarvis says: {response}")
                    speak_text(response)

        except Exception as e:
            print("An error occurred: {}".format(e))

