from writansub.types import Segment, Transcript, WordInfo


def seg(start, end, text, prob=0.9, time_kind="estimate"):
    return Segment(start=start, end=end, text=text, time_kind=time_kind, words=[WordInfo(text, prob)])


def transcript(*specs, backend="fake"):
    segments = [seg(*s) for s in specs]
    duration = segments[-1].end if segments else 0.0
    return Transcript(segments=segments, duration=duration, backend=backend)


class FakeRecognizer:
    name = "fake"

    def __init__(self, full=None, replies=()):
        self.full = full if full is not None else transcript()
        self._replies = iter(replies)
        self.calls: list[dict] = []

    def tag(self):
        return "fake"

    def cache_key(self):
        return "fake:test"

    def load(self, device):
        return object()

    def transcribe(self, model, path, lang, progress=None, *, short_clip=False):
        self.calls.append({"path": path, "lang": lang, "short_clip": short_clip, "progress": progress})
        if not short_clip:
            return self.full
        answer = next(self._replies)
        if isinstance(answer, Exception):
            raise answer
        return answer() if callable(answer) else answer
