import argparse
import io
import json
import os
import random
import token
from io import BytesIO
from tokenize import tokenize

import pandas as pd
from tqdm.auto import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-input_dir', type=str, required=True)
arg_parser.add_argument('-output_dir', type=str, required=True)
args = arg_parser.parse_args()

TRAIN_FILE = os.path.join(args.input_dir, 'conala-train.json')
TEST_FILE = os.path.join(args.input_dir, 'conala-test.json')

OUT_TRAIN_FILE = os.path.join(args.output_dir, 'train.json')
OUT_TEST_FILE = os.path.join(args.output_dir, 'test.json')
OUT_DEV_FILE = os.path.join(args.output_dir, 'dev.json')

PUNCTUATION = {
    'keep'  : "&,",
    'remove': u'\u200b' + '?!`™•°'
}


def clean_text(x):
    x = x.lower()

    for p in PUNCTUATION['keep']:
        x = x.replace(p, "%s" % p)
    for p in PUNCTUATION['remove']:
        x = x.replace(p, "")

    return x


def do_tokenize(code):
    tok_list = list(tokenize(io.BytesIO(code.encode('utf-8')).readline))
    out = []

    for t in tok_list:
        if token.tok_name[t.type] in ['ENCODING', 'NEWLINE', 'ENDMARKER', 'INDENT', 'DEDENT']:
            continue

        if all([not t.string.isspace(), t.string != '']):
            out.append(t.string)

    return out


class Sketch(object):
    def __init__(self, init):
        self.token_list = None
        self.type_list = None

        if init is not None:
            if isinstance(init, list):
                self.set_by_list(init, None)
            elif isinstance(init, tuple):
                self.set_by_list(init[0], init[1])
            elif isinstance(init, str):
                self.set_by_str(init)
            else:
                raise NotImplementedError

    def set_by_str(self, f):
        tk_list = list(tokenize(BytesIO(f.strip().encode('utf-8')).readline))[1:-1]
        self.token_list = [tk.string for tk in tk_list]
        self.type_list = [token.tok_name[tk.type] for tk in tk_list]

    # well-tokenized token list
    def set_by_list(self, token_list, type_list):
        self.token_list = list(token_list)
        if type_list is not None:
            self.type_list = list(type_list)

    def to_list(self):
        return self.token_list

    def __str__(self):
        return ' '.join(self.to_list())

    def layout(self):
        assert len(self.token_list) == len(self.type_list)

        r_list = []

        for tk, tp in zip(self.token_list, self.type_list):
            if tp in ('NEWLINE', 'INDENT', 'DEDENT'): continue
            r_list.append(tp)

        return r_list


def preprocess(df):
    out_data = []

    for i, ex in tqdm(df.iterrows(), total=len(df)):
        label = ex['rewritten_intent'] if ex['rewritten_intent'] is not None else ex['intent']
        label = [x.strip() for x in list(map(clean_text, label.split())) if x.strip() != '']

        code = do_tokenize(ex['snippet'])

        sketch = Sketch(ex['snippet']).layout()

        assert len(sketch) == len(code), "sketch [%s] != code [%s]" % (str(sketch), str(code))

        out_data.append({'token': code, 'src': label, 'type': sketch})

    return out_data


if __name__ == '__main__':
    train_data = preprocess(df=pd.read_json(TRAIN_FILE))

    random.shuffle(train_data)
    n = int(0.1 * len(train_data))
    dev_data, train_data = train_data[:n], train_data[n:]

    test_data = preprocess(df=pd.read_json(TEST_FILE))

    with open(OUT_TRAIN_FILE, "wt") as fp:
        for d in train_data:
            fp.write(json.dumps(d) + "\n")

    with open(OUT_TEST_FILE, "wt") as fp:
        for d in test_data:
            fp.write(json.dumps(d) + "\n")

    with open(OUT_DEV_FILE, "wt") as fp:
        for d in dev_data:
            fp.write(json.dumps(d) + "\n")
