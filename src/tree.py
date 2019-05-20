import token
from io import BytesIO
from tokenize import tokenize

SKP_WORD = '<sk>'
RIG_WORD = '<]>'
LFT_WORD = '<[>'


class SketchRepresentation(object):
    def __init__(self, init):
        self.code_token_list = None
        self.sketch_token_list = None

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
        self.code_token_list = [tk.string for tk in tk_list]
        self.sketch_token_list = [token.tok_name[tk.type] for tk in tk_list]

    # well-tokenized token list
    def set_by_list(self, code_token_list, sketch_token_list):
        self.code_token_list = list(code_token_list)
        if sketch_token_list is not None:
            self.sketch_token_list = list(sketch_token_list)

    def to_list(self):
        return self.code_token_list

    def __str__(self):
        return ' '.join(self.to_list())

    def layout(self, add_skip=False):
        assert len(self.code_token_list) == len(self.sketch_token_list)

        r_list = []
        for code_tok, sketch_tok in zip(self.code_token_list, self.sketch_token_list):
            if sketch_tok in ['OP', 'KEYWORD']:
                r_list.append(code_tok)
            elif sketch_tok in ['STRING']:
                if add_skip:
                    s_list = code_tok.split(' ')
                    r_list.extend([LFT_WORD] + [SKP_WORD for __ in range(len(s_list) - 2)] + [RIG_WORD])
                else:
                    r_list.append(sketch_tok)
            else:
                r_list.append(sketch_tok)

        return r_list

    def target(self):
        assert len(self.code_token_list) == len(self.sketch_token_list)

        r_list = []
        for code_tok, sketch_tok in zip(self.code_token_list, self.sketch_token_list):
            if sketch_tok in ['STRING']:
                s_list = code_tok.split(' ')
                r_list.extend([LFT_WORD] + s_list[1:-1] + [RIG_WORD])
            else:
                r_list.append(code_tok)

        return r_list

    def norm(self, not_layout=False):
        return self


def is_code_eq(tokens1, tokens2, not_layout=False):
    # TODO: make smarter

    if isinstance(tokens1, SketchRepresentation):
        tokens1 = str(tokens1)
    else:
        tokens1 = ' '.join(tokens1)

    if isinstance(tokens2, SketchRepresentation):
        tokens2 = str(tokens2)
    else:
        tokens2 = ' '.join(tokens2)

    tokens1 = ['\"' if it in (RIG_WORD, LFT_WORD) else it for it in tokens1.split(' ')]
    tokens2 = ['\"' if it in (RIG_WORD, LFT_WORD) else it for it in tokens2.split(' ')]

    if len(tokens1) != len(tokens2):
        return False

    return all(map(lambda tk1, tk2: tk1 == tk2, tokens1, tokens2))


if __name__ == '__main__':
    s = "1 if True else 0".split()
    t = SketchRepresentation((s, "NUMBER NAME NAME NAME NUMBER".split()))
    print(1, t)
    print(2, t.to_list())
    print(3, ' '.join(t.layout(add_skip=False)))
    print('\n')
