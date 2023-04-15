import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import functools
from unidecode import unidecode

class Preprocessor():

    def __init__(self, remove_num):
        self.stopwords = set(stopwords.words('english'))
        self.remove_num = remove_num
        self.stemmer = SnowballStemmer(language='english')

    def trim(self, text):
        text = text.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
        return text

    def remove_punctuation(self, term):
        if self.remove_num:
            x = unidecode(''.join(character for character in term if character not in string.punctuation and not(character.isdigit())))
        else:
            x = unidecode(''.join(character for character in term if character not in string.punctuation))
        return x

    def tokenise(self, text):
        tokens = re.split('\W+', text)
        lower_text = list(map(str.lower, tokens))
        return list(map(self.remove_punctuation, lower_text))


    def remove_stopwords(self, terms):
        return [term for term in terms if term not in self.stopwords]

    @functools.lru_cache()
    def stem_term(self, term):
        return self.stemmer.stem(term)
  
    def stem(self, terms: list):
        return [self.stem_term(term) for term in terms]

    def remove_empty(self, terms):
        return [term for term in terms if term != '']

    def preprocess(self, raw_text):
        
        text = self.trim(raw_text)
        tokens = self.tokenise(text)
        words_without_ST = self.remove_stopwords(tokens)
        stemmed = self.stem(words_without_ST)
        no_empties = self.remove_empty(stemmed)
        return no_empties
    




# %%

