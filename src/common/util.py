import re
import spacy

nlp = spacy.load('en_core_web_lg')
stop_words = nlp.Defaults.stop_words


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def merge_sentences(sentences):
    text = ''
    for row in sentences:
        text += row +" "
    text = text[:-1]
    return text


def get_lemma(text):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return lemmas


def remove_stop_word(tokens):
   return [word for word in tokens if not word in stop_words]


def combine(row):
    headline = row[2]
    body = row[3]
    stop_word = True
    headline_lema = get_lemma(clean(headline))
    body_lema = get_lemma(clean(body))
    if stop_word:
        headline_lema = remove_stop_word(headline_lema)
        body_lema = remove_stop_word(body_lema)
    combine = headline_lema + body_lema
    return combine, headline_lema, body_lema