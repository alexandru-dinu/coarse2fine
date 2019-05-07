from mittens.tf_mittens import Mittens
import numpy as np
import pickle
import csv


CO_OCCURRENCE_FILE 	= '../data_model/comp-sci-corpus-thr20000-window10.mat'
VOCAB_FILE 			= '../data_model/comp-sci-corpus-thr20000-window10.vocab'
EMB_GLOVE_FILE		= '../data_model/glove.840B.300d.txt'


def load_glove(emb_glove_file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(emb_glove_file, encoding='latin'))    
    return embeddings_index


def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed


def closest_to(emb_glove, w, n=1):
    xs = []
    
    for w_ in tqdm(emb_glove):
        if w == w_: continue
        xs += [(w_, np.dot(emb_glove[w], emb_glove[w_])/(np.linalg.norm(emb_glove[w]) * np.linalg.norm(emb_glove[w_])))]

    return [x for x, _ in sorted(xs, key=lambda x:-x[1])[:n]]


def fine_tune():
	co_occurrence = pickle.load(open(CO_OCCURRENCE_FILE, "rb"))
	print("Loaded co-occurrence")

	vocab = pickle.load(open(VOCAB_FILE, "rb"))
	print("Loaded vocab")

	emb_glove = glove2dict(EMB_GLOVE_FILE)
	print("Loaded glove")

	mittens_model = Mittens(n=300, max_iter=1000)

	new_emb_glove = mittens_model.fit(
    	co_occurrence,
    	vocab=list(vocab.keys()),
    	initial_embedding_dict=emb_glove
	)

	return new_emb_glove


if __name__ == '__main__':
	new_emb = fine_tune()

	pickle.dump(new_emb, open('../data_model/glove-fine-tuned', 'wb'))