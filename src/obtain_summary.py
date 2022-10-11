
import argparse
import pandas as pd
import os
# Text Rank summarizer
from common.Text_rank import TextRank
# To load BART summarizer
from transformers import pipeline
# BERT summarizer
from summarizer import Summarizer
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from tqdm import tqdm


def main(parser):
    args = parser.parse_args()
    dataset_in = args.dataset_in
    dataset_out = args.dataset_out
    summary_type = args.summary_type
    exec_summary(dataset_in, dataset_out, summary_type)


def exec_summary(dataset_in, dataset_out, summary_type):
    df = pd.read_json(os.getcwd() + dataset_in, lines=True)

    # Only FNC dataset
    df_groupby_article = df.groupby(["Id_Article", "sentences2"])
    print(list(df_groupby_article.groups.keys()))
    group_df = pd.DataFrame(df_groupby_article.groups.keys(), columns=["Id_Article", "sentences2"])

    summaries = []
    if summary_type == 'text_rank':
        tr = TextRank('english', 5)
        n_threads = cpu_count() - 4
        with Pool(n_threads) as p:
            summaries = list(tqdm(p.imap(tr.cal_summary, group_df['sentences2']), total=len(group_df)))
    elif summary_type == 'bart':
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0, batch_size=8)
        for summary in tqdm(summarizer(list(group_df["sentences2"]), max_length=350, min_length=170, truncation=True)):
            summaries.append(summary['summary_text'])
    elif summary_type == 'bert':
        summarizer = Summarizer()
        for value in tqdm(list(group_df["sentences2"]), total=len(group_df)):
          summary_value = summarizer(value, num_sentences=5)
          if not summary_value:
             summary_value = value
          summaries.append(summary_value)

    # Only for FNC
    dict_id_summary = dict(zip(list(group_df["Id_Article"]), summaries))

    df = df.drop(columns=['sentences2'])
    df['sentences2'] = df["Id_Article"].apply(lambda x: dict_id_summary[x])

    df.to_json(os.getcwd() + dataset_out, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--dataset_in",
                        default="/data/FNC_body_train.json",
                        type=str,
                        help="This parameter is the relative dir of dataset.")

    parser.add_argument("--dataset_out",
                        default="/data/FNC_summary_train.json",
                        type=str,
                        help="This parameter is the relative dir of preprocessed dataset.")

    parser.add_argument("--summary_type",
                        default="text_rank",
                        type=str,
                        help="This parameter is used for choose type of summary, (text_rank, bert, bart).")
    main(parser)