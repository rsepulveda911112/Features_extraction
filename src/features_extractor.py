import argparse
import os
from multiprocessing import cpu_count, Pool
import json
import time

import numpy as np
import pandas as pd
import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from common.similarityMetrics import SimilarityMetrics
from common.TFIDF import TFIDF
from common.util import combine
from common.overlap import word_overlap_feature
from common.bert_cosine import SentenceSimilarity
from common.soft_cosine_similarity import SoftCosineSimilarity


nltk.download('vader_lexicon')


sip = SentimentIntensityAnalyzer()


def main(parser):
    args = parser.parse_args()
    dataset_in = args.dataset_in
    dataset_out = args.dataset_out
    start_time = time.time()
    exec_preprocessing(dataset_in, dataset_out)
    print(time.time() - start_time)


def exec_preprocessing(dataset_in, dataset_out):
    file_in = os.getcwd() + dataset_in
    datas = pd.read_json(file_in, lines=True)
    df = pd.DataFrame(datas)

    n_threads = cpu_count() - 4

    headlines_lemas = []
    bodies_lemas = []
    heads_bodies_combine = []

    with Pool(n_threads) as p:
        proccess = p.imap(combine, df.itertuples(name=None))
        for head_body_combine, headline_lema, body_lema in tqdm.tqdm(proccess, total=len(df)):
            headlines_lemas.append(headline_lema)
            bodies_lemas.append(body_lema)
            heads_bodies_combine.append(head_body_combine)


    similarityMetrics = SimilarityMetrics('/resource/model_lda_preprocess', heads_bodies_combine)
    tf_idf = TFIDF('/resource/tfidf.dat', df['sentence1'], df['sentences2'])

    sentencesSimilarity = SentenceSimilarity(device=0)
    sentencesSimilarity.encoded_text(list(df['sentence1']), list(df['sentences2']))
    softCosineSimilarity = SoftCosineSimilarity()

    values_similarity = [sentencesSimilarity.cosine_similarity(sent1, sent2) for sent1, sent2 in
                         zip(df['sentence1'], df['sentences2'])]

    values = []
    for (index, row), headline_lema, body_lema, bert_cosine \
            in tqdm.tqdm(zip(df.iterrows(), headlines_lemas, bodies_lemas, values_similarity), total=len(df)):
        values.append(cal_metrics(row, headline_lema, body_lema, bert_cosine, similarityMetrics, tf_idf, softCosineSimilarity))

    file_out = open(os.getcwd() + dataset_out, "w+")
    for row in tqdm.tqdm(values, total=len(values)):
        file_out.write(json.dumps(row) + "\n")


def cal_metrics(row, headline_lema, summary_lema, bert_cosine, similarityMetrics, tf_idf, softCosineSimilarity):
    headline = row['sentence1']
    summary = row['sentences2']
    overlap_feature = word_overlap_feature(headline, summary, False)
    cosine_similarity_value = tf_idf.get_cosine_similarity(headline, summary)
    soft_cosine_similarity_value = softCosineSimilarity.calculate(headline_lema, summary_lema)
    hellingerScore, jaccardScore, kullbackScore = similarityMetrics.all_metrics(headline_lema, summary_lema)
    kullbackScore = np.float64(kullbackScore)

    ############################   Polarity ########################33
    polarityClaim = sip.polarity_scores(headline)
    polarityBody = sip.polarity_scores(summary)

    return {'Id_Article': row['Id_Article'], 'sentence1': headline,
            'sentences2': summary, 'label': row['label'],
            'cosine_similarity': cosine_similarity_value,
            'jaccard_distance': jaccardScore, 'hellinger_distance': hellingerScore,
            'kullback_leibler_distance': kullbackScore, 'overlap': overlap_feature,
            'bert_cosine_similarity': bert_cosine,
            'soft_cosine_similarity': soft_cosine_similarity_value,
            'polarityClaim_nltk_neg': polarityClaim['neg'],
            'polarityClaim_nltk_pos': polarityClaim['pos'],
            'polarityClaim_nltk_neu': polarityClaim['neu'],
            'polarityClaim_nltk_compoud': polarityClaim['compound'],
            'polarityBody_nltk_neg': polarityBody['neg'],
            'polarityBody_nltk_pos': polarityBody['pos'],
            'polarityBody_nltk_neu': polarityBody['neu'],
            'polarityBody_nltk_compoud': polarityBody['compound']
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--dataset_in",
                        default="/data/FNC_body_train.json",
                        type=str,
                        help="This parameter is the relative dir of dataset.")

    parser.add_argument("--dataset_out",
                        default="/data/FNC_summary_train_features.json",
                        type=str,
                        help="This parameter is the relative dir of preprocessed dataset.")
    main(parser)