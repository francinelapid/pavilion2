import re
from sre_parse import Pattern
import sys
if sys.version_info[:2] >= (3,5):
    import typing


class TokenizeError(ValueError):
    def __init__(self, msg, seq_stub, pos):
        self.args = msg
        self.msg = msg
        self.seq_stub = seq_stub
        self.pos = pos

    def __str__(self):

        return "{s.msg} at pos {s.pos}: {stub}"\
               .format(s=self, stub=self.seq_stub[self.pos:20])

class ParseError(ValueError):
    PREVIEW_LIMIT = 20

    def __init__(self, msg, tokens):
        """Denotes an error in parsing.
        :param msg: The error message.
        :param tokens: A selection of tokens starting at where the error
            occurred.
        """
        self.args = msg
        self.tokens = tokens

    def __str__(self):

        preview = []
        for tok in self.tokens:
            preview.append(str(tok))

            if sum([len(p) for p in preview]) > self.PREVIEW_LIMIT:
                break

        preview = ''.join(preview)

        return "{}: {}".format(self.msg, preview)


T_SYNTAX = 'T_SYNTAX'
T_CHILD = 'T_CHILD'
T_OP = 'T_OP'


class BaseToken:

    # type: Pattern
    REGEX = None

    TYPE = T_CHILD

    def __init__(self, pos, value, raw, whitespace):
        self.pos = pos
        self._set_value(value)
        self.raw = raw
        self.whitespace = whitespace

    def _set_value(self, values):
        if len(values) == 1:
            self.value = values[0]
        elif not values:
            self.value = None
        else:
            self.value = values

    @classmethod
    def match(cls, seq, pos, whitespace):
        """
        :param seq:
        :param pos:
        :return:
        """

        match = cls.REGEX.match(seq, pos)
        print("matching {} '{}' {}"
              .format(cls.__name__, seq[pos:20], match))

        if match is None:
            return None

        value = match.groups()
        raw = match.group()

        token = cls(pos, value, raw, whitespace)

        return token

    def __str__(self):
        return self.whitespace + self.raw


def new_token(name, regex, tok_type=T_CHILD):
    return type(
        name,
        (BaseToken,),
        {
            'REGEX': re.compile(regex),
            'TYPE': tok_type
        }
    )


class ParserRuleSet:

    WHITESPACE = ' \n\r\t'
    PATTERNS = {}
    IGNORE_WHITESPACE = True

    TYPE = T_CHILD

    def __init__(self, pattern_id, *args):
        self.pattern_id = pattern_id
        self._set_args(args)

        self.args = args
        self.child = None
        self.children = []
        self.op = None
        self.ops = []

    def _set_args(self, *args):
        """Assign the args by type.
        :param args:
        :return:
        """

        children = []
        ops = []

        for arg in args:
            if arg.TYPE == T_CHILD:
                children.append(arg)
            elif arg.TYPE == T_OP:
                ops.append(arg)

        if children:
            self.child = children[0]
            self.children = children
        if ops:
            self.op = ops[0]
            self.ops = ops

    @classmethod
    def add_pattern(cls, *pattern, name=None):
        """
        :param str name: The name of this pattern.
        :param Tuple[Union[BaseToken, TokenPattern]] pattern: A possible sequence
            of tokens supported by this
        :return:
        """

        if name is None:
            name = len(cls.PATTERNS)

        # Doing type checking only to help the user define BaseToken Patterns in
        # a sensible way.
        if not all([issubclass(t, (BaseToken, ParserRuleSet)) for t in pattern]):
            raise ValueError("TokenPattern patterns must be a list or tuple"
                             "of Tokens or TokenPatterns. Got {} for pattern "
                             "{}."
                             .format(pattern, name))

        cls.PATTERNS[name] = pattern

    @classmethod
    def all_token_types(cls, seen=None):
        """Return all possible tokens for this token pattern (for parsing).
        :rtype: List[BaseToken]

        """

        tokens = set()

        if seen is None:
            seen = set()

        print(cls.__name__, cls.PATTERNS)

        for pattern_id in cls.PATTERNS.keys():
            pattern = cls.PATTERNS[pattern_id]
            for tok_type in pattern:
                if tok_type in seen:
                    continue
                elif issubclass(tok_type, ParserRuleSet):
                    seen.add(tok_type)
                    tokens = tokens.union(tok_type.all_token_types(seen))
                else:
                    tokens.add(tok_type)
        print(cls.__name__, tokens)

        return tokens

    @classmethod
    def tokenize(cls, seq, pos=0):
        """
        :param str seq:
        :param int pos:
        :return:
        """

        token_types = cls.all_token_types()
        tokens = []

        while pos < len(seq):

            whitespace = []
            if cls.IGNORE_WHITESPACE:
                while seq and seq[pos] in cls.WHITESPACE:
                    whitespace.append(seq[pos])
                    pos += 1
            whitespace = ''.join(whitespace)

            for tok_type in token_types:

                tok = tok_type.match(seq, pos, whitespace)

                if tok is not None:
                    tokens.append(tok)
                    pos += len(tok.raw)
                    break
            else:
                raise TokenizeError("Unmatched sequence.", seq, pos)

        return tokens

    @classmethod
    def first_set(cls):

        seen_rules = set()

        tokens = set()

        for pattern in cls.PATTERNS:
            first_item = pattern[0]
            if issubclass(first_item, ParserRuleSet):
                if first_item in seen_rules:
                    pass
                else:
                    item_tokens = first_item.first_tokens()
                    seen_rules.add(first_item)
                    tokens.append()

    @classmethod
    def references(cls):
        """Return a the set of parser rule sets reachable from this one."""

        rules = set()
        unsearched = [cls]
        while unsearched:
            rule_set = unsearched.pop(0)
            if rule_set is not cls:
                rules.add(rule_set)

            for pattern in rule_set.PATTERNS.values():
                for tr in pattern:
                    if issubclass(tr, BaseToken):
                        pass
                    elif (tr not in rules and
                          tr not in unsearched and
                          tr is not cls):
                        unsearched.append(tr)
                        rules.add(tr)
        return rules

    @classmethod
    def parse(cls, tokens, seen_states=None):
        """
        :param [BaseToken] tokens:
        :return:
        :rtype: (ParserRuleSet, int)
        """

        if seen_states is None:
            seen_states = []
        else:
            seen_states = seen_states.copy()

        # We've looped back around to the same state.
        if cls in seen_states:
            return None, None

        seen_states.append(cls)

        for pattern_id in cls.PATTERNS.keys():
            pattern = cls.PATTERNS[pattern_id]

            print('parsing {cls.__name__} pattern {pattern}'
                  .format(cls=cls, pattern=pattern))

            rule_args = []
            pos = 0
            for tok_type in pattern:
                print('tok_type {tt.__name__}, {t}'
                      .format(tt=tok_type, t=tokens[pos]))
                if issubclass(tok_type, BaseToken):
                    if isinstance(tokens[pos], tok_type):
                        rule_args.append(tokens[pos])
                        pos += 1
                        # Reset the seen states to just this class now that
                        # we've seen a matching token.
                        seen_states = [cls]
                    else:
                        print('huh')
                        break
                # It must be a ParserRuleSet
                else:
                    rule, pos_offset = tok_type.parse(
                        tokens[pos:],
                        seen_states=seen_states)
                    rule_args.append(rule)
                    pos += pos_offset
                    if rule is None:
                        # Try the next one.
                        break
            else:
                # The rule matched
                return cls(pattern_id, *rule_args), pos
        else:
            return None, None

    def print(self, tab_level=0):
        print("<{}>".format(self.__class__.__name__))

        self._print(tab_level)

    def _print(self, tab_level):
        if len(self.ops) == 1 and len(self.children) == 1:
            print("  "*tab_level, end="")
            print("{} {}".format(self.ops[0], self.children[0]))
        elif len()


    def blarg(self):
        if len(cls.PATTERNS) != 1 or len(cls.PATTERNS[0]) != 1:
            raise ValueError("Invalid CFG. The start rule must have "
                             "a single transition to a single non-terminal "
                             "rule.")

        if cls.PATTERNS[0] is cls:
            raise ValueError(
                "Invalid CFG. The Start rule is recursive."
            )

        rule_0_refs = cls.PATTERNS[0].references()
        if cls.PATTERNS[0] in rule_0_refs:
            raise ValueError(
                "Invalid CFG. The rule N, where Start->N, must not be "
                "reachable from itself.")

        rules = []
        for rule_set in cls.references():
            for pattern in rule_set.PATTERNS:
                rules.append((rule_set, pattern))
        rules.append((cls, cls.PATTERNS[0]))
        unstated_rules = rules.copy()

        states = {}
        rule_id = 0
        transitions = [(None,) + cls.PATTERNS[0]]

        while transitions:
            trans = transitions.pop(0)

            targets = [trans]
            .
            if issubclass(trans[-1], ParserRuleSet):
                ended_trans = trans[-1]

    @classmethod
    def _parse(cls, tokens):

        states = {}

        rules = cls.references()

        print(rules)

        raise ValueError














def new_rule(name):
    """
    :param name:
    :return:
    :rtype: ParserRuleSet
    """

    return type(
        name,
        (ParserRuleSet,),
        {
            'PATTERNS': {},
        }
    )


if __name__ == '__main__':

    Number = new_token('Number', r'\d+')
    Plus = new_token('Plus', r'\+', tok_type=T_OP)
    Minus = new_token('Minus', r'\-', tok_type=T_OP)
    Multiply = new_token('Multiply', r'\*', tok_type=T_OP)
    Divide = new_token('Divide', r'\/', tok_type=T_OP)
    OpenParen = new_token('OpenParen', r'\(', tok_type=T_SYNTAX)
    CloseParen = new_token('CloseParen', r'\)', tok_type=T_SYNTAX)

    Start = new_rule('Start')
    AExpr = new_rule('AExpr')
    MExpr = new_rule('MExpr')
    BExpr = new_rule('BExpr')

    Start.add_pattern(AExpr)

    AExpr.add_pattern(AExpr, Plus, MExpr, name='add')
    AExpr.add_pattern(AExpr, Minus, MExpr, name='subtract')
    AExpr.add_pattern(MExpr)

    MExpr.add_pattern(MExpr, Multiply, BExpr, name='multiply')
    MExpr.add_pattern(MExpr, Divide, BExpr, name='divide')
    MExpr.add_pattern(BExpr)

    BExpr.add_pattern(Number)
    BExpr.add_pattern(OpenParen, AExpr, CloseParen)

    tokens = Start.tokenize('1 + 2 +3')
    print(tokens)
    Start.parse(tokens)
