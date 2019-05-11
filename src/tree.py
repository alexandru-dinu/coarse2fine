import token
from io import BytesIO
from tokenize import tokenize

SKP_WORD = '<sk>'
RIG_WORD = '<]>'
LFT_WORD = '<[>'


class SketchRepresentation(object):
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

    def layout(self, add_skip=False):
        assert len(self.token_list) == len(self.type_list)
        r_list = []
        for tk, tp in zip(self.token_list, self.type_list):
            if tp in ('OP', 'KEYWORD'):
                r_list.append(tk)
            elif tp in ('STRING',):
                if add_skip:
                    s_list = tk.split(' ')
                    r_list.extend(
                        [LFT_WORD] + [SKP_WORD for __ in range(len(s_list) - 2)] + [RIG_WORD])
                else:
                    r_list.append(tp)
            # elif tp in ('NAME', 'NUMBER'):
            #     if add_skip:
            #         r_list.append(SKP_WORD)
            #     else:
            #         r_list.append(tp)
            else:
                r_list.append(tp)
        return r_list

    def target(self):
        assert len(self.token_list) == len(self.type_list)
        r_list = []
        for tk, tp in zip(self.token_list, self.type_list):
            if tp in ('STRING',):
                s_list = tk.split(' ')
                r_list.extend([LFT_WORD] + s_list[1:-1] + [RIG_WORD])
            else:
                r_list.append(tk)
        return r_list

    def norm(self, not_layout=False):
        return self


def is_code_eq(t1, t2, not_layout=False):
    if isinstance(t1, SketchRepresentation):
        t1 = str(t1)
    else:
        t1 = ' '.join(t1)
    if isinstance(t2, SketchRepresentation):
        t2 = str(t2)
    else:
        t2 = ' '.join(t2)

    t1 = ['\"' if it in (RIG_WORD, LFT_WORD) else it for it in t1.split(' ')]
    t2 = ['\"' if it in (RIG_WORD, LFT_WORD) else it for it in t2.split(' ')]

    if len(t1) != len(t2):
        return False

    return all(map(lambda tk1, tk2: tk1 == tk2, t1, t2))


if __name__ == '__main__':
    s = "1 if True else 0".split()
    t = SketchRepresentation((s, "NUMBER NAME NAME NAME NUMBER".split()))
    print(1, t)
    print(2, t.to_list())
    print(3, ' '.join(t.layout(add_skip=False)))
    print('\n')
