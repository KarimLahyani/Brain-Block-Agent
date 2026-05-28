from collections import Counter

from brainblock_diverse.pieces import board_to_text


class SolutionMemory:

    def __init__(self, repeat_penalty=4.0, similarity_penalty=2.0, similarity_threshold=0.85):
        self.repeat_penalty = repeat_penalty
        self.similarity_penalty = similarity_penalty
        self.similarity_threshold = similarity_threshold
        self.counts = Counter()
        self.text_by_signature = {}

    def __len__(self):
        return len(self.counts)

    def similarity(self, sig1, sig2):
        same = sum(a == b for a, b in zip(sig1, sig2))
        return same / len(sig1)

    def max_similarity(self, signature):
        if not self.counts:
            return 0.0
        return max(self.similarity(signature, old_sig) for old_sig in self.counts.keys())

    def penalty_for(self, signature):
        if signature in self.counts:
            return self.repeat_penalty, False, 1.0

        sim = self.max_similarity(signature)
        if sim <= self.similarity_threshold:
            return 0.0, True, sim

        ratio = (sim - self.similarity_threshold) / (1.0 - self.similarity_threshold)
        return self.similarity_penalty * ratio, True, sim

    def register(self, signature, board):
        self.counts[signature] += 1
        if signature not in self.text_by_signature:
            self.text_by_signature[signature] = board_to_text(board)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, (signature, count) in enumerate(self.counts.items(), start=1):
            lines.append(f"Solution {i} | seen {count} time(s)")
            lines.append(self.text_by_signature[signature])
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
