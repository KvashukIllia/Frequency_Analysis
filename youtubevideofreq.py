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

def get_frequency(text, start_year = 2018, end_year = 2019, smoothing = 0, corpus = "en-US-2019"):
    words = clean_text(text)[0]
    text_df = clean_text(text)[1]
    words = urllib.parse.quote(words)
    url = 'https://books.google.com/ngrams/json?content=' + words + '&year_start=' + str(start_year) + '&year_end=' + str(end_year) + '&corpus=' + str(corpus) + '&smoothing=' + str(smoothing) + '' 
    response = requests.get(url) 
    outputs = response.json() 
    freq = {}
    freq = {output['ngram']: np.max([i for i in output['timeseries'] if i!=0]) for output in outputs}
    df = pd.DataFrame(freq.items(), columns= ["Word", "Language Frequency"])
    sorted_df = df.sort_values(by = "Language Frequency", ascending=False)
    sorted_df["Language Order"] = range(0, len(df))
    combined_text_language_df = sorted_df.join(text_df.set_index("Word"), on = 'Word')
    return combined_text_language_df.set_index('Word')

def clean_text(words):
    pattern_cleaning = r'[{}]'.format(re.escape(string.punctuation))
    cleaned_text = re.sub(pattern_cleaning, " ", words)
    cleaned_text = cleaned_text.lower().split()
    cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
    unique_words = set(cleaned_text)
    string_unique_words = ",".join(unique_words)
    sample_size = len(cleaned_text)
    words_df = pd.DataFrame([(word, cleaned_text.count(word)/sample_size) for word in unique_words], columns= ['Word', 'Text Frequency'])
    words_df = words_df.sort_values(by = "Text Frequency")
    words_df["Order"] = range(0, len(words_df))
    return string_unique_words, words_df

if __name__ == '__main__':
    link = str(sys.argv[1])
    text = get_transcript(link)
    feq = get_frequency(text)
    print(feq.to_string())
