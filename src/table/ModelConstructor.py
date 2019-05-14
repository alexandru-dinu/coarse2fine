import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchtext.vocab
from tqdm.auto import tqdm

import table
import table.Models
import table.modules
from table.Models import CopyGenerator, LayCoAttention, ParserModel, QCoAttention, RNNEncoder, SeqDecoder
from table.modules.Embeddings import PartUpdateEmbedding


def load_orig_glove(emb_file: str) -> dict:
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if os.path.isfile(emb_file + ".pickle"):
        print(" * load glove from pickle")
        emb = pickle.load(open(emb_file + ".pickle", "rb"))
    else:
        print(" * load glove from txt, dumping to pickle")
        emb = dict(get_coefs(*o.split(" ")) for o in open(emb_file, encoding='latin'))
        pickle.dump(emb, open(emb_file + ".pickle", "wb"))

    return emb


def load_glove_fine_tuned(args, get_only_dict=False):
    """
    :return: return dict if get_only_dict is True, otherwise return Vectors
    """

    _cache = os.path.dirname(args.word_embeddings)
    _name = os.path.basename(args.word_embeddings)

    assert os.path.join(_cache, _name) == args.word_embeddings

    print(" * load vocab from [%s]" % args.vocab_file)
    print(" * load pre-trained glove from [%s]" % args.pt_embeddings)
    print(" * load fine-tuned glove from [%s]" % args.word_embeddings)

    glove_emb = load_orig_glove(args.pt_embeddings)

    ft_glove_emb_arr = pickle.load(open(args.word_embeddings, "rb"))
    vocab = pickle.load(open(args.vocab_file, "rb"))

    ft_glove_emb = {w: ft_glove_emb_arr[i] for w, i in vocab.items()}

    for w in tqdm(ft_glove_emb, desc="Mixing embeddings: (%.2f ft, %.2f pt)" % (args.ft_factor, args.pt_factor)):
        if w not in glove_emb:
            glove_emb[w] = ft_glove_emb[w]
        else:
            glove_emb[w] = args.ft_factor * ft_glove_emb[w] + args.pt_factor * glove_emb[w]

    if get_only_dict:
        print(" * returning emb dict")
        return glove_emb

    else:
        print(" * returning torchtext.vocab.Vectors")

        # save emb_dict as .pt
        itos = list(glove_emb.keys())
        stoi = {}
        vectors = {}

        for i, w in tqdm(enumerate(glove_emb), total=len(glove_emb), desc="Construct stoi and vectors"):
            stoi[w] = i
            vectors[i] = torch.FloatTensor(glove_emb[w])

        dim = len(vectors[0])
        torch.save([itos, stoi, vectors, dim], os.path.join(_cache, _name + ".pt"))

        # len(vocab) x dim
        return torchtext.vocab.Vectors(name=_name, cache=_cache)


def make_word_embeddings(args, vocab: torchtext.vocab.Vocab):
    word_padding_idx = vocab.stoi[table.IO.PAD_WORD]
    num_word = len(vocab)
    emb_word = nn.Embedding(num_word, args.word_emb_size, padding_idx=word_padding_idx)

    print(" * using embeddings [%s]" % args.word_embeddings)

    if args.word_embeddings != '':  # TODO: might get rid of this check?

        # load custom embeddings
        if args.use_custom_embeddings:
            print(" * loading custom embeddings")
            vocab.load_vectors(vectors=[load_glove_fine_tuned(args, get_only_dict=False)])
            emb_word.weight.data.copy_(vocab.vectors)

        else:
            print(" * using default embeddings")

            if args.word_emb_size == 150:
                dim_list = ['100', '50']
            elif args.word_emb_size == 250:
                dim_list = ['200', '50']
            else:
                dim_list = [str(args.word_emb_size), ]

            vectors = [torchtext.vocab.GloVe(name="6B", cache=args.word_embeddings, dim=it) for it in dim_list]
            vocab.load_vectors(vectors)
            emb_word.weight.data.copy_(vocab.vectors)
    # ---

    if args.fix_word_vecs:
        # <unk> is 0
        num_special = len(table.IO.SPECIAL_TOKEN_LIST)
        # zero vectors in the fixed embedding (emb_word)
        emb_word.weight.data[:num_special].zero_()
        emb_special = nn.Embedding(num_special, args.word_emb_size, padding_idx=word_padding_idx)
        emb = PartUpdateEmbedding(num_special, emb_special, emb_word)
        return emb
    else:
        return emb_word


def make_embeddings(word_dict, vec_size):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    w_embeddings = nn.Embedding(num_word, vec_size, padding_idx=word_padding_idx)
    return w_embeddings


def make_encoder(args, embeddings, ent_embedding=None):
    return RNNEncoder(
        args.rnn_type, args.brnn, args.enc_layers, args.rnn_size,
        args.dropout, args.dropout_i, args.lock_dropout, args.dropword_enc,
        args.weight_dropout, embeddings, ent_embedding
    )


def make_layout_encoder(args, embeddings):
    return RNNEncoder(
        args.rnn_type, args.brnn, args.enc_layers, args.decoder_input_size,
        args.dropout, args.dropout_i, args.lock_dropout, args.dropword_enc,
        args.weight_dropout, embeddings, ent_embedding=None
    )


def make_q_co_attention(args):
    if args.q_co_attention:
        return QCoAttention(
            args.rnn_type, args.brnn, args.enc_layers, args.rnn_size,
            args.decoder_input_size, args.dropout, args.weight_dropout, 'dot', args.attn_hidden
        )

    return None


def make_lay_co_attention(args):
    if args.lay_co_attention:
        return LayCoAttention(
            args.rnn_type, args.brnn, args.enc_layers, args.decoder_input_size,
            args.rnn_size, args.dropout, args.weight_dropout, 'mlp', args.attn_hidden
        )

    return None


def make_decoder(args, fields, field_name, embeddings, input_size):
    decoder = SeqDecoder(
        args.rnn_type, args.brnn, args.dec_layers, embeddings,
        input_size, args.rnn_size, args.global_attention, args.attn_hidden,
        args.dropout, args.dropout_i, args.lock_dropout, args.dropword_dec, args.weight_dropout
    )

    if field_name == 'tgt':
        classifier = CopyGenerator(
            args.dropout, args.rnn_size, args.rnn_size,
            fields['tgt'].vocab, fields['copy_to_ext'].vocab, args.copy_prb
        )
    else:
        classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_size, len(fields[field_name].vocab)),
            nn.LogSoftmax()
        )

    return decoder, classifier


def make_base_model(model_args, fields, checkpoint=None):
    print(" * make word embeddings")
    w_embeddings = make_word_embeddings(model_args, vocab=fields["src"].vocab)

    if model_args.ent_vec_size > 0:
        ent_embedding = make_embeddings(fields["ent"].vocab, model_args.ent_vec_size)
    else:
        ent_embedding = None

    print(" * make question encoder")
    q_encoder = make_encoder(model_args, w_embeddings, ent_embedding)
    if model_args.separate_encoder:
        q_tgt_encoder = make_encoder(model_args, w_embeddings, ent_embedding)
        q_encoder = (q_encoder, q_tgt_encoder)

    if model_args.layout_token_prune:
        w_token_embeddings = make_word_embeddings(model_args, fields["src"].vocab)
        q_token_encoder = make_encoder(model_args, w_token_embeddings, ent_embedding)
        token_pruner = nn.Sequential(
            nn.Dropout(model_args.dropout),
            nn.Linear(model_args.rnn_size, len(fields['lay'].vocab) - len(table.IO.SPECIAL_TOKEN_LIST))  # skip special tokens
        )
    else:
        q_token_encoder = None
        token_pruner = None

    print(" * make layout decoder models")
    lay_field = 'lay'
    lay_embeddings = make_embeddings(fields[lay_field].vocab, model_args.decoder_input_size)
    lay_decoder, lay_classifier = make_decoder(model_args, fields, lay_field, lay_embeddings, model_args.decoder_input_size)

    print(" * make target decoder models")
    if model_args.no_share_emb_layout_encoder:
        lay_encoder_embeddings = make_embeddings(
            fields[lay_field].vocab, model_args.decoder_input_size)
    else:
        lay_encoder_embeddings = lay_embeddings
    if model_args.no_lay_encoder:
        lay_encoder = lay_embeddings
    else:
        lay_encoder = make_layout_encoder(model_args, lay_encoder_embeddings)

    q_co_attention = make_q_co_attention(model_args)
    lay_co_attention = make_lay_co_attention(model_args)

    tgt_embeddings = make_embeddings(fields['tgt'].vocab, model_args.decoder_input_size)
    tgt_decoder, tgt_classifier = make_decoder(model_args, fields, 'tgt', None, model_args.decoder_input_size)

    print(" * make ParserModel")
    model = ParserModel(
        q_encoder, q_token_encoder, token_pruner, lay_decoder, lay_classifier,
        lay_encoder, q_co_attention, lay_co_attention, tgt_embeddings,
        tgt_decoder, tgt_classifier, model_args
    )

    if checkpoint is not None:
        print(' * loading model from checkpoint [%s]' % model_args.model_path)
        model.load_state_dict(checkpoint['model'])

    if model_args.cuda:
        print(" * put on cuda")
        model.cuda()
    else:
        print(" * put on cpu")

    return model
