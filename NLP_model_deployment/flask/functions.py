import pickle
import re
import numpy as np
import contractions
import emoji
import time
import string 
from nltk.corpus import stopwords, wordnet

class PreProcess:
    def __init__(self):
        pass

    # Remove emojis 
    def remove_emojis(self, reviews):
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF" 
            u"\U00002500-\U00002BEF"  
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  
            u"\u3030"
                        "]+", re.UNICODE)
        return re.sub(emoj, '', reviews)

    def remove_stopwords(self, reviews):
        STOPWORDS = stopwords.words('english')
        STOPWORDS.remove('not')
        STOPWORDS.remove('is')
        STOPWORDS.remove('but')
        if STOPWORDS is None:
            STOPWORDS = set(stopwords.words('english'))
        return ' '.join([word for word in reviews.split() if word not in STOPWORDS])

    def remove_extra_whitespace(self, reviews):
        return " ".join(reviews.split())

    def lower_case(self, reviews):
        reviews = reviews.lower()
        return reviews

    def change_contractions(self, reviews):
        expanded_words = [contractions.fix(word) for word in reviews.split()]
        expanded_review = ' '.join(expanded_words)
        return expanded_review

    # Remove Punctuations
    def remove_punctuations(self, reviews):
        new_review = reviews.translate(str.maketrans('', '', string.punctuation))
        return new_review
    # Remove numbers
    def remove_numbers(self, reviews):
        mapping = str.maketrans('', '', string.digits)
        new_review = reviews.translate(mapping)
        return new_review

    def clean_text(self, reviews):
        reviews = self.lower_case(reviews)
        reviews = self.change_contractions(reviews)
        reviews = self.remove_emojis(reviews)
        reviews = self.remove_punctuations(reviews)
        reviews = self.remove_numbers(reviews)

        return reviews