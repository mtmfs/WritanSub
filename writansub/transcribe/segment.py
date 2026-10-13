import unicodedata

from writansub.types import Segment, WordInfo

SENTENCE_END = frozenset("。！？!?")
CLAUSE_END = frozenset("、，,")
PARTICLES = frozenset(
    "は が を に で と の へ も か ね よ な ば や から まで より って でも ても ては には では とは とか "
    "し ので のに けど けれど ながら つつ たり だ です ます ました でした た て ない ん".split()
)
CLAUSE_BONUS = 0.3
BALANCE_WEIGHT = 0.6

Token = tuple[str, float, float]


def _kept(ch: str) -> bool:
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def attach_punctuation(text: str, tokens: list[Token]) -> list[Token]:
    kept = [i for i, ch in enumerate(text) if _kept(ch)]
    if not tokens or "".join(text[i] for i in kept) != "".join(t for t, _, _ in tokens):
        return tokens
    out: list[Token] = []
    k = 0
    for tok, start, end in tokens:
        first = kept[k]
        k += len(tok)
        stop = kept[k] if k < len(kept) else len(text)
        out.append((text[first:stop].strip(), start, end))
    return out


def _join(parts: list[str]) -> str:
    out = ""
    for p in parts:
        if out and p and out[-1].isascii() and out[-1].isalnum() and p[0].isascii() and p[0].isalnum():
            out += " "
        out += p
    return out


def _core(tok: str) -> str:
    return "".join(ch for ch in tok if _kept(ch))


def _chars(piece: list[Token]) -> int:
    return sum(len(t) for t, _, _ in piece)


def _best_cut(piece: list[Token]) -> int | None:
    total = _chars(piece)
    best: int | None = None
    best_score = 0.0
    left = 0
    for i in range(1, len(piece)):
        left += len(piece[i - 1][0])
        if _core(piece[i][0]) in PARTICLES:
            continue
        pause = max(0.0, piece[i][1] - piece[i - 1][2])
        bonus = CLAUSE_BONUS if piece[i - 1][0][-1] in CLAUSE_END else 0.0
        imbalance = abs(2 * left - total) / total
        score = pause + bonus - BALANCE_WEIGHT * imbalance
        if best is None or score > best_score:
            best, best_score = i, score
    return best


def _split_long(piece: list[Token], max_chars: int) -> list[list[Token]]:
    if _chars(piece) <= max_chars or len(piece) < 2:
        return [piece]
    cut = _best_cut(piece)
    if cut is None:
        return [piece]
    return _split_long(piece[:cut], max_chars) + _split_long(piece[cut:], max_chars)


def split_tokens(tokens: list[Token], max_gap: float = 0.8, max_chars: int = 40,
                 time_kind: str = "estimate") -> list[Segment]:
    pieces: list[list[Token]] = []
    group: list[Token] = []
    for tok in tokens:
        text, start, _end = tok
        if not text:
            continue
        if group and start - group[-1][2] >= max_gap:
            pieces.append(group)
            group = []
        group.append(tok)
        if text[-1] in SENTENCE_END:
            pieces.append(group)
            group = []
    if group:
        pieces.append(group)

    segments: list[Segment] = []
    for piece in pieces:
        for chunk in _split_long(piece, max_chars):
            segments.append(Segment(
                start=chunk[0][1],
                end=chunk[-1][2],
                text=_join([t for t, _, _ in chunk]),
                time_kind=time_kind,
                words=[WordInfo(t, None, s, e) for t, s, e in chunk],
            ))
    return segments
