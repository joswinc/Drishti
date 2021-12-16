import speech_recognition as sr
import pocketsphinx
import os
import sys
import re
import webbrowser
import smtplib
import requests
import subprocess
from pyowm import OWM
import youtube_dl
import vlc
import urllib
import json
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
import wikipedia
import random
from time import strftime
from face import face_store,face_recog
from ocr import ocr_core
from pocketsphinx import LiveSpeech, get_model_path
import cv2
from img_cap.main import main,imgcaptest

model,sess,data,vocabulary = main()


model_path = get_model_path()
#print(model_path)

r = sr.Recognizer()
with sr.Microphone() as source:
        print('Please Wait....Sensing Ambient Noise!')
        #r.pause_threshold =  1
        r.adjust_for_ambient_noise(source, duration=2)

def subCommand():
    "listens for commands"
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Say something...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio).lower()
        print('You said: ' + command + '\n')
    #loop back to continue to listen for commands if unrecognizable speech is received
    except sr.UnknownValueError:
        print('....')
        command = myCommand()
    return command


def AIResponse(audio):
    "speaks audio passed as argument"
    newaudio=""
    if audio!='':
        if audio[0]=='b':
            audio=audio[1:]
    for char in audio:
        if char!='\"' and char!="'" and char!="\\":
            newaudio+=char
    print(newaudio)
    for line in newaudio.splitlines():
        os.system("espeak '" + line+"'")


def myCommand():
    "listens for commands"
    model_path = get_model_path()
    #print(model_path)
    try:
        speech = LiveSpeech(
            verbose=False,
            sampling_rate=16000,
            buffer_size=2048,
            no_search=False,
            full_utt=False,
            hmm=os.path.join(model_path, 'en-us'),
            lm='./speech/5747.lm',
            dic='./speech/5747.dic'
        )
    except:
        print("Error occured in LiveSpeech")
        speech=''
    for phrase in speech:
        print(str(phrase).lower())
        print(len(str(phrase).lower()))
        #print(str(phrase).lower())
        try:
            assistant(str(phrase).lower())
        except:
            print("Error occured in commands!")
        return


def assistant(command):
    "if statements for executing commands"
    print(command)
    if 'shutdown' == command:
        AIResponse('Bye bye Sir. Have a nice day')
        #sess.close()
        sys.exit()

    
#greetings
    elif 'hello' == command:
        day_time = int(strftime('%H'))
        if day_time < 12:
            AIResponse('Hello Sir. Good morning')
        elif 12 <= day_time < 18:
            AIResponse('Hello Sir. Good afternoon')
        else:
            AIResponse('Hello Sir. Good evening')
    elif 'help me' == command:
    
        AIResponse("""
        You can use these commands and I'll help you out:
        1. Can you tell the weather : Tells you the current condition and temperture
        2. Can you tell me the news : Tells you the news in India
        3. What is the time : Tells you the system time
        4. Wikipedia Search : Gives you information from wikipedia
        5. Save this face : Saves the face image for face recognition
        6. Who is this : Tells you the persons name 
        7. What is in front of me : Tells you what is in front of you
        """)
        
    
#top stories from google news
    elif 'news' in command and 'tell' in command:
        try:
            news_url="https://news.google.com/news/rss"
            Client=urlopen(news_url)
            xml_page=Client.read()
            Client.close()
            soup_page=soup(xml_page,"xml")
            news_list=soup_page.findAll("item")
            for news in news_list[:5]:
                AIResponse(str(news.title.text.encode('utf-8')))
        except Exception as e:
                print(e)
#current weather
    elif 'the weather' in command and 'tell' in command:
        AIResponse('Which City')
        scommand = subCommand()
        try:
            reg_ex = re.search('(.*)', str(scommand))
            if reg_ex:
                city = reg_ex.group(1)
                owm = OWM(API_key='ab0d5e80e8dafb2cb81fa9e82431c1fa')
                obs = owm.weather_at_place(city)
                w = obs.get_weather()
                k = w.get_status()
                x = w.get_temperature(unit='celsius')
                AIResponse('Current weather in %s is %s. The maximum temperature is %0.2f and the minimum temperature is %0.2f degree celcius' % (city, k, x['temp_max'], x['temp_min']))
        except Exception as e:
            print(e)
       
#time
    elif 'what is the time' in command or 'what is the current time' in command:
        import datetime
        now = datetime.datetime.now()
        AIResponse('Current time is %d hours %d minutes' % (now.hour, now.minute))

#askme anything
    elif 'wikipedia search' == command:
        AIResponse('Want do you want to search in wikipedia')
        scommand=subCommand()
        reg_ex = re.search('about (.*)', scommand)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                ny = wikipedia.page(topic)
                AIResponse("\""+str(ny.content[:500].encode('utf-8'))+"\"")
        except Exception as e:
                print(e)
                AIResponse(str(e))
#face save
    elif 'save this face' in command:
        AIResponse('Tell a name')
        scommand = subCommand()
        reply = face_store(str(scommand))
        AIResponse(reply)

    elif 'who is this' in command:
        face_names = face_recog()
        if len(face_names)>0:
            for face in face_names:
                AIResponse(face)
            	

    elif 'read this' in command:
        reply = ocr_core()
        AIResponse(reply)

    elif 'what is in front of me' in command or 'describe view' in command:
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()
        cv2.destroyAllWindows()
        cv2.imwrite('./img_cap/test/images/image.jpg',frame)
        imgcaptest(model,sess,data,vocabulary)
        #os.system("cd img_cap ;python3 ./main.py --phase=test --model_file='./models/289999.npy' --beam_size=3" )
    else:
        print("-")

        


AIResponse('Hi User, I am Drishti and I am your personal voice assistant, Please give a command or say "help me" and I will tell you what all I can do for you.')
#loop to continue executing multiple commands
#AIResponse('Hi')
while True:
    myCommand()
