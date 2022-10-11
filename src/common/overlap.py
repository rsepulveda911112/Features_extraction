from src.common.util import clean, get_lemma


# def word_overlap_features(headlines, bodies, stopWord):
#     X = []
#     for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):     
#         feature = len(set(headline).intersection(body)) / float(len(set(headline).union(body)))
#         X.append(feature)
#     return X

def word_overlap_feature(headline, body, stopWord):
    headline = get_lemma(clean(headline))
    body = get_lemma(clean(body))
    return len(set(headline).intersection(body)) / float(len(set(headline).union(body)))