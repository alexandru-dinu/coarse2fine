from collections import defaultdict

from tree import is_code_eq


class ParseResult(object):
    def __init__(self, idx, lay, tgt, token_prune):
        self.idx = idx
        self.lay = lay
        self.tgt = tgt
        self.token_prune = token_prune

        self.correct = defaultdict(lambda: 0)
        self.incorrect = defaultdict(lambda: [])

        self.incorrect_prune = set()

    def eval(self, gold):
        if is_code_eq(self.lay, gold['lay'], not_layout=False):
            self.correct['lay'] = 1
        else:
            self.incorrect['lay'].append((self.lay, gold['lay']))

        if is_code_eq(self.tgt, gold['tgt'], not_layout=True):
            self.correct['tgt'] = 1
        else:
            self.incorrect['tgt'].append((self.tgt, gold['tgt']))

    def get_by_name(self, name):
        if name in ['tgt', 'tgt-token']:
            return self.tgt

        if name in ['lay', 'lay-token']:
            return self.lay

        return None

    def __str__(self):
        return "ParseResult:\n\tidx: %s\n\tlay: %s\n\ttgt = %s" % (self.idx, self.lay, self.tgt)

    def __repr__(self):
        return str(self)
