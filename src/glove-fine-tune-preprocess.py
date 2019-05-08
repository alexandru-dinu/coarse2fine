#!/usr/bin/env python
# coding: utf-8


import os
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

stopWords = set(stopwords.words('english'))

import numpy as np
import re

PUNCTUATION = {
    'sep'   : u'\u200b' + "/-'´′‘…—−–",
    'keep'  : "&",
    'remove': '?!.,，"#$%\'()*+-/:;<=>@[\\]^_`{|}~“”’™•°'
}


def clean_text(x):
    x = x.lower()

    for p in PUNCTUATION['sep']:
        x = x.replace(p, " ")
    for p in PUNCTUATION['keep']:
        x = x.replace(p, " %s " % p)
    for p in PUNCTUATION['remove']:
        x = x.replace(p, "")

    return x


def do_tfidf(directory: str, vocabulary_size: int):
    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files]

    tfidf = TfidfVectorizer(
        input="filename",
        decode_error="ignore",
        analyzer="word",
        stop_words="english",
        token_pattern='[a-zA-Z]+',
        max_features=vocabulary_size,
        # min_df=0.01
    )

    tfidf.fit(files)

    return tfidf.get_feature_names()


base_dir = '../data_model/comp-sci-corpus/'


def get_all_words(base_dir):
    all_words = []

    for f in tqdm(os.listdir(base_dir)):
        file_contents = [clean_text(l.strip().lower()) for l in open(os.path.join(base_dir, f), "rt").readlines()]

        for line in file_contents:
            for w in line.split():
                if re.match(r'[\w]+', w) and w not in stopWords:
                    all_words.append(w)

    print("len(all_words) = %d" % len(all_words))
    return all_words


#
# vocab = Counter()
# for w in tqdm(all_words):
#     vocab[w] += 1
#
# print("len(vocab) = %d" % len(vocab))


thr = 20000
window = 10

# top_words, top_freqs = zip(*vocab.most_common(thr))
# top_words = set(top_words)

print("TF-DIF")
top_words = set(do_tfidf(base_dir, vocabulary_size=thr))
print("len = ", len(top_words))

word2idx = {w: i for i, w in enumerate(top_words)}

M = np.zeros((thr, thr), dtype=np.uint16)

all_words = get_all_words(base_dir)

print("Constructing co-occurrence matrix")
for i in tqdm(range(len(all_words))):
    if all_words[i] not in top_words:
        continue

    for j in range(max(i - window, 0), min(i + window, len(all_words))):
        if i == j or all_words[j] not in top_words: continue

        M[word2idx[all_words[i]], word2idx[all_words[j]]] += 1

out_vocab_file = '../data_model/comp-sci-corpus-thr%d-window%d-tfidf.vocab' % (thr, window)
out_mat_file = '../data_model/comp-sci-corpus-thr%d-window%d-tfidf.mat' % (thr, window)

pickle.dump(word2idx, open(out_vocab_file, "wb"))
pickle.dump(M, open(out_mat_file, "wb"))
