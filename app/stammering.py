from __future__ import annotations

import re
from collections import Counter


class StammeringDetector:
    """Detect stammering in a translated sentence."""

    # Minimum score to consider a translation as stammering
    SCORE_THRESHOLD = 1.0

    # N-gram sizes to analyse
    NGRAM_SIZES = [1, 2, 3, 4, 5, 6]

    # An n-gram repeated more than this many times is suspicious
    NGRAM_REPEAT_LIMIT = {1: 4, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2}

    # Character window sizes for sub-word repetition
    CHAR_WINDOW_SIZES = [10, 15, 20, 30, 40]
    CHAR_REPEAT_LIMIT = 2.5   # appearing more than this -> suspicious

    # Unique-token ratio below this -> suspicious
    UNIQUE_RATIO_THRESHOLD = 0.45

    # Translation / source length ratio above this -> suspicious
    LENGTH_RATIO_THRESHOLD = 3.5

    # Public API
    def detect(self,
               source_sentence: str,
               translated_sentence: str) -> bool:
        """Return True if the translated sentence appears to contain stammering."""

        score = self._score(source_sentence, translated_sentence)
        return score >= self.SCORE_THRESHOLD
    
    def _source_repetition_discount(self, source: str) -> float:
        src_tokens = self._tokenise(source)
        if len(src_tokens) < 2:
            return 1.0
        
        src_ratio = len(set(src_tokens)) / len(src_tokens)
        # If source is at least as repetitive as the translation would be,
        # the translation is just doing its job, no stammer
        if src_ratio < self.UNIQUE_RATIO_THRESHOLD:
            return 0.0  # fully suppress repetition signals
        return 1.0

    # Internal scoring
    def _score(self, source: str, translation: str) -> float:
        score = 0.0
        tokens = self._tokenise(translation)
        if len(tokens) == 0:
            return 0.0

        # Discount repetition signals when source is itself repetitive
        repetition_discount = self._source_repetition_discount(source)

        score += self._ngram_signal(tokens) * repetition_discount
        score += self._char_repeat_signal(translation) * repetition_discount
        score += self._unique_ratio_signal(tokens) * repetition_discount
        score += self._consecutive_duplicate_signal(tokens) * repetition_discount
        score += self._length_anomaly_signal(source, translation)
        return score

    def _ngram_signal(self, tokens: list[str]) -> float:
        """Score based on over-represented n-grams."""

        signal = 0.0
        n_tokens = len(tokens)
        for n in self.NGRAM_SIZES:
            if n_tokens < n:
                continue
            ngrams = [tuple(tokens[i:i + n]) for i in range(n_tokens - n + 1)]
            counts = Counter(ngrams)
            limit = self.NGRAM_REPEAT_LIMIT.get(n, 2)
            for ngram, cnt in counts.items():
                if cnt > limit:
                    # Weight larger n-grams more heavily (more suspicious)
                    weight = 0.3 + 0.1 * n
                    # Scale by how much over the limit we are
                    excess = cnt - limit
                    signal += weight * excess
        return min(signal, 3.0)  # cap contribution

    def _char_repeat_signal(self, text: str) -> float:
        """Detect repeated sub-strings at character level."""

        signal = 0.0
        clean = text.lower()
        for w in self.CHAR_WINDOW_SIZES:
            if len(clean) < w * 2:
                continue
            windows = [clean[i:i + w] for i in range(len(clean) - w + 1)]
            counts = Counter(windows)
            for phrase, cnt in counts.items():
                if phrase.strip() and cnt > self.CHAR_REPEAT_LIMIT:
                    signal += 0.4 * (cnt - self.CHAR_REPEAT_LIMIT)
        return min(signal, 2.0)

    def _unique_ratio_signal(self, tokens: list[str]) -> float:
        """Low unique-token ratio indicates heavy repetition."""

        if len(tokens) < 6:
            return 0.0
        ratio = len(set(tokens)) / len(tokens)
        if ratio < self.UNIQUE_RATIO_THRESHOLD:
            # The lower the ratio, the stronger the signal
            return (self.UNIQUE_RATIO_THRESHOLD - ratio) * 4.0
        return 0.0

    def _consecutive_duplicate_signal(self, tokens: list[str]) -> float:
        """Consecutive identical single tokens or bigrams are almost always a stammer."""

        signal = 0.0
        # Single token duplicates
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1] and len(tokens[i]) > 1:
                # repetition penalty
                signal += 0.3
        # Bigram duplicates
        for i in range(len(tokens) - 3):
            if tokens[i:i + 2] == tokens[i + 2:i + 4]:
                signal += 0.8
        return min(signal, 3.0)

    def _length_anomaly_signal(self,
                               source: str,
                               translation: str) -> float:
        """Flag if translation is disproportionately longer than source."""

        src_len = max(len(source.split()), 1)
        tgt_len = len(translation.split())
        ratio = tgt_len / src_len
        if ratio > self.LENGTH_RATIO_THRESHOLD:
            return min((ratio - self.LENGTH_RATIO_THRESHOLD) * 0.3, 1.5)
        return 0.0

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        return [t for t in text.split() if t]