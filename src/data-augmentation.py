import os
import pickle
import random

import nltk
import numpy as np
from nltk.corpus import wordnet

VOCAB_FILE = '../data_model/comp-sci-corpus-thr20000-window10.vocab'
EMB_GLOVE_FILE = '../data_model/glove.840B.300d.txt'
FT_EMB_FILE = '../data_model/glove-fine-tuned-5000'


def load_glove() -> dict:
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if os.path.isfile(EMB_GLOVE_FILE + ".pickle"):
        emb = pickle.load(open(EMB_GLOVE_FILE + ".pickle"), "rb")
    else:
        emb = dict(get_coefs(*o.split(" ")) for o in open(EMB_GLOVE_FILE, encoding='latin'))
        pickle.dump(emb, open(EMB_GLOVE_FILE + ".pickle", "wb"))

    return emb


def load_glove_fine_tuned() -> dict:
    ft_emb_arr = pickle.load(open(FT_EMB_FILE, "rb"))
    vocab = pickle.load(open(VOCAB_FILE, "rb"))

    # len(vocab) x 300
    return {w: ft_emb_arr[i] for w, i in vocab.items()}


def closest_to(emb: dict, w: str, n=1):
    xs = []

    for w_ in emb:
        if w == w_:
            continue
        xs += [(w_, np.dot(emb[w], emb[w_]) / (np.linalg.norm(emb[w]) * np.linalg.norm(emb[w_])))]

    return [x for x, _ in sorted(xs, key=lambda e: -e[1])[:n]]


def get_similar_words(word: str, emb: dict):
    return closest_to(emb, word.lower(), n=10)


def substitute(word: str, emb: dict):
    # get similar words
    similar_words = set(get_similar_words(word, emb))

    # get synonyms
    synonyms = wordnet.synsets(word)
    synonyms = set([syn.lemmas()[0].name().lower() for syn in synonyms])

    print("Word: [%s]" % word)
    print("\tSimilar words (from emb):", similar_words)
    print("\tSynonyms (from wordnet):", synonyms)
    print()

    # get intersection
    intersection = synonyms.intersection(similar_words)

    if len(intersection) == 0:
        return word

    return random.choice(list(intersection))


def augment_text(text: str, emb: dict):
    text = nltk.word_tokenize(text)
    word_pos_tag = nltk.pos_tag(text)

    for i in range(len(word_pos_tag)):
        word, pos_tag = word_pos_tag[i]

        # check if is noun (singular or plural) or verb
        if pos_tag in ['NN', 'NNS'] or pos_tag.startswith("VB"):
            text[i] = substitute(word, emb)

    return " ".join(text)


def main():
    emb = load_glove_fine_tuned()
    # emb = load_glove()

    text = "call the function and assign the result to variable x"
    aug = augment_text(text, emb)

    print("Original:", text)
    print("Augmented:", aug)


if __name__ == "__main__":
    main()
