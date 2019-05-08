import os
import random

import nltk
import numpy as np
from nltk.corpus import wordnet

import argparse

from table.ModelConstructor import load_orig_glove, load_glove_fine_tuned

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-data_model_dir", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data_model')
)
arg_parser.add_argument("-vocab_file", type=str)  # e.g. comp-sci-corpus-thr20000-window10.vocab
arg_parser.add_argument("-emb_file", type=str)  # e.g. glove.840B.300d.txt
arg_parser.add_argument("-ft_emb_file", type=str)  # e.g. glove-fine-tuned-tfidf-2000

args = arg_parser.parse_args()


# args.vocab_file = os.path.join(args.data_model_dir, 'comp-sci-corpus-thr20000-window10.vocab')
# args.emb_file = os.path.join(args.data_model_dir, 'glove.840B.300d.txt')
# args.ft_emb_file = os.path.join(args.data_model_dir, 'glove-fine-tuned-tfidf-2000')


def closest_to(emb: dict, w: str, n=1):
    xs = []

    for w_ in emb:
        if w == w_:
            continue
        xs += [(w_, np.dot(emb[w], emb[w_]) / (np.linalg.norm(emb[w]) * np.linalg.norm(emb[w_])))]

    return [x for x, _ in sorted(xs, key=lambda e: -e[1])[:n]]


def get_similar_words(word: str, emb: dict):
    return closest_to(emb, word.lower(), n=20)


def substitute(word: str, emb: dict):
    # get similar words
    similar_words = set(get_similar_words(word, emb))

    # get synonyms
    synonyms = wordnet.synsets(word)
    synonyms = set([syn.lemmas()[0].name().lower() for syn in synonyms])

    # get intersection
    intersection = synonyms.intersection(similar_words)

    print("Word: [%s]" % word)
    print("\temb:", similar_words)
    print("\twordnet:", synonyms)
    print("\tintersection: ", intersection)
    print()

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
            print("> substituting word %s" % word)
            text[i] = substitute(word, emb)

    return " ".join(text)


def main():
    emb = load_glove_fine_tuned(args.ft_emb_file, get_only_dict=True)

    text = "call the function and assign the result to variable x"
    aug = augment_text(text, emb)

    print("Original:", text)
    print("Augmented:", aug)


if __name__ == "__main__":
    main()
