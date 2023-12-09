import sys
import numpy as np
import pandas as pd
import requests 
import urllib
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.corpus import stopwords
# nltk.download('stopwords')

PATTERN = "watch\?v=(.*)"
pd.options.display.float_format = '{:.10f}'.format


def get_transcript(link):
    video_id = re.findall(pattern= PATTERN, string = link)[0]
    srt = YouTubeTranscriptApi.get_transcript(video_id, 
                                          languages=['en'])
    transcript_lines = [timestamp['text'] for timestamp in srt]
    transcript_merged = " ".join(transcript_lines)
    return transcript_merged


def clean_text(words):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    pattern_cleaning = r'[{}]'.format(re.escape(string.punctuation))
    cleaned_text = re.sub(pattern_cleaning, " ", words)
    cleaned_text = cleaned_text.lower().split()
    cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
    cleaned_text = [word for word in cleaned_text if not word in stop_words]
    return cleaned_text
    
def preprocess_cleaned_text(cleaned_text):
    unique_words = set(cleaned_text)
    string_unique_words = ",".join(unique_words)
    sample_size = len(cleaned_text)
    words_df = pd.DataFrame([(word, cleaned_text.count(word)/sample_size) for word in unique_words], columns= ['Word', 'Text Frequency'])
    words_df = words_df.sort_values(by = "Text Frequency")
    words_df["Text Order"] = range(0, len(words_df))
    return string_unique_words, words_df

def get_frequency(text, start_year = 2018, end_year = 2019, smoothing = 0, corpus = "en-US-2019"):
    words, text_df = preprocess_cleaned_text(clean_text(text))
    words = urllib.parse.quote(words)
    url = (f'https://books.google.com/ngrams/json?content={words}' 
           f'&year_start= {start_year}'
           f'&year_end= {end_year}'
           f'&corpus={corpus}'
           f'&smoothing= {smoothing}') 
    response = requests.get(url) 
    outputs = response.json() 
    scale_factor = 10**8
    freq = {output['ngram']: np.mean([i for i in output['timeseries'] if i!=0]) for output in outputs}
    df = pd.DataFrame(freq.items(), columns= ["Word", "Language Frequency"])
    sorted_df = df.sort_values(by = "Language Frequency", ascending=False)
    sorted_df["Language Order"] = range(0, len(df))
    combined_text_language_df = sorted_df.join(text_df.set_index("Word"), on = 'Word')
    combined_text_language_df["Order Offset"] = combined_text_language_df["Language Order"] - combined_text_language_df["Text Order"]
    combined_text_language_df = combined_text_language_df.set_index('Word')
    return combined_text_language_df[["Language Frequency", "Text Frequency", "Language Order", "Text Order", "Order Offset"]]

if __name__ == '__main__':
    link = str(sys.argv[1])
    text = get_transcript(link)
    feq = get_frequency(text)
    print(feq.to_string())
