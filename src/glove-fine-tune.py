import pickle
from pprint import pprint

import numpy as np
from mittens.tf_mittens import Mittens
from tqdm.auto import tqdm

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--test", action="store_true", help="Evaluating")
arg_parser.add_argument("--num_iter", type=int, default=1000, help="Number of training iterations")
args = arg_parser.parse_args()

CO_OCCURRENCE_FILE = '../data_model/comp-sci-corpus-thr20000-window10.mat'
VOCAB_FILE = '../data_model/comp-sci-corpus-thr20000-window10.vocab'
EMB_GLOVE_FILE = '../data_model/glove.840B.300d.txt'
FT_EMB_FILE = '../data_model/glove-fine-tuned'


def load_glove():
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMB_GLOVE_FILE, encoding='latin'))

    return embeddings_index


def load_glove_fine_tuned() -> dict:
    ft_emb_arr = pickle.load(open(FT_EMB_FILE + "-%d" % args.num_iter, "rb"))
    vocab = pickle.load(open(VOCAB_FILE, "rb"))

    # len(vocab) x 300
    return {w: ft_emb_arr[i] for w, i in vocab.items()}


def closest_to(emb: dict, w, n=1):
    xs = []

    for w_ in tqdm(emb):
        if w == w_:
            continue
        xs += [(w_, np.dot(emb[w], emb[w_]) / (np.linalg.norm(emb[w]) * np.linalg.norm(emb[w_])))]

    return [x for x, _ in sorted(xs, key=lambda x: -x[1])[:n]]


def fine_tune(num_iter=1000) -> None:
    print("Loading co-occurrence [%s]" % CO_OCCURRENCE_FILE)
    co_occurrence = pickle.load(open(CO_OCCURRENCE_FILE, "rb"))

    print("Loading vocab [%s]" % VOCAB_FILE)
    vocab = pickle.load(open(VOCAB_FILE, "rb"))

    print("Loading glove [%s]" % EMB_GLOVE_FILE)
    emb_glove = load_glove()

    print("Training for %d iterations" % num_iter)
    mittens_model = Mittens(n=300, max_iter=num_iter)
    new_emb_glove = mittens_model.fit(
        co_occurrence,
        vocab=list(vocab.keys()),
        initial_embedding_dict=emb_glove
    )

    print("Done. Saving to file [%s]" % (FT_EMB_FILE + "-%d" % num_iter))
    pickle.dump(new_emb_glove, open(FT_EMB_FILE + "-%d" % num_iter, 'wb'))


def do_test():
    # print("Loading glove [%s]" % EMB_GLOVE_FILE)
    # emb = load_glove()

    print("Loading ft-glove [%s]" % (FT_EMB_FILE + "-%d" % args.num_iter))
    ft_emb = load_glove_fine_tuned()

    while True:
        word = input("> enter word: ").strip()

        # print("original emb")
        # try:
        #     pprint(closest_to(emb, word, n=20))
        # except KeyError:
        #     print("%s not in emb" % word)
        #
        # print()

        print("fine-tuned emb")
        try:
            pprint(closest_to(ft_emb, word, n=20))
        except KeyError:
            print("%s not in ft_emb" % word, "\n")


if __name__ == '__main__':
    if args.test:
        do_test()
    else:
        fine_tune(num_iter=args.num_iter)
