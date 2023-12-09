import sys
import numpy as np
import pandas as pd
import requests 
import urllib
import re
import string
from nltk.stem import WordNetLemmatizer
from youtube_transcript_api import YouTubeTranscriptApi

pattern = "watch\?v=(.*)"
lemmatizer = WordNetLemmatizer()

def get_transcript(link):
    video_id = re.findall(pattern= pattern, string = link)[0]
    srt = YouTubeTranscriptApi.get_transcript(video_id, 
                                          languages=['en'])
    transcript_lines = [timestamp['text'] for timestamp in srt]
    transcript_merged = " ".join(transcript_lines)
    return transcript_merged

def get_frequency(words, start_year = 2018, end_year = 2019, smoothing = 0, corpus = "en-US-2019"):
    words = urllib.parse.quote(words)
    url = 'https://books.google.com/ngrams/json?content=' + words + '&year_start=' + str(start_year) + '&year_end=' + str(end_year) + '&corpus=' + str(corpus) + '&smoothing=' + str(smoothing) + '' 
    response = requests.get(url) 
    outputs = response.json() 
    freq = {}
    freq = {output['ngram']: np.max([i for i in output['timeseries'] if i!=0]) for output in outputs}
    df = pd.DataFrame(freq.items(), columns= ["Word", "Frequency"])
    return df.sort_values(by = "Frequency", ascending=False)

def clean_text(words):
    pattern = r'[{}]'.format(re.escape(string.punctuation))
    cleaned_text = re.sub(pattern, " ", words)
    cleaned_text = cleaned_text.lower().split()
    cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
    cleaned_text = set(cleaned_text)
    cleaned_text = ",".join(cleaned_text)
    return cleaned_text

if __name__ == '__main__':
    link = str(sys.argv[1])
    text = get_transcript(link)
    cleaned_text = clean_text(text)
    feq = get_frequency(cleaned_text)
    print(feq.to_string())
