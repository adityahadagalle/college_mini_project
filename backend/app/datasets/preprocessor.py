"""Text preprocessing module for dataset cleaning with negation preservation."""

import re
import unicodedata
from typing import List, Set

from app.core.dataset_exceptions import PreprocessingError
from app.datasets.config import DatasetConfig

# Explicit whitelist of negation and sentiment-critical tokens that must NEVER be stripped
NEGATION_WHITELIST: Set[str] = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "nothing",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "isn't",
    "isnt",
    "doesn't",
    "doesnt",
    "aren't",
    "arent",
    "won't",
    "wont",
    "cant",
    "can't",
    "cannot",
    "couldn't",
    "couldnt",
    "shouldn't",
    "shouldnt",
    "wouldn't",
    "wouldnt",
    "didn't",
    "didnt",
    "don't",
    "dont",
    "haven't",
    "havent",
    "hasn't",
    "hasnt",
    "hadn't",
    "hadnt",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "ain't",
    "aint",
    "bad",
    "poor",
}


class TextPreprocessor:
    """Configurable text cleaning and normalization pipeline with negation preservation."""

    def __init__(self, config: DatasetConfig):
        """Initialize preprocessor with dataset configuration.

        Args:
            config (DatasetConfig): Preprocessing configuration settings.
        """
        self.config = config
        self._compile_regexes()

    def _compile_regexes(self) -> None:
        """Compile regex patterns for high-performance text cleaning."""
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27ff]", flags=re.UNICODE)
        self.whitespace_pattern = re.compile(r"\s+")
        # Preserve words, whitespace, sentiment punctuation, and apostrophes/hyphens for contractions
        self.non_sentiment_punct = re.compile(r"[^\w\s!?:;.,\'\"-]")

    def clean_text(self, text: str) -> str:
        """Clean a single string using configured preprocessing rules.

        Args:
            text (str): Input raw text.

        Returns:
            str: Cleaned text.

        Raises:
            PreprocessingError: If cleaning fails due to invalid inputs.
        """
        if not isinstance(text, str):
            raise PreprocessingError(f"Expected string input, received {type(text).__name__}")

        try:
            cleaned = text

            if self.config.normalize_unicode:
                cleaned = unicodedata.normalize("NFKD", cleaned)

            if self.config.remove_html_tags:
                cleaned = self.html_pattern.sub(" ", cleaned)

            if self.config.remove_urls:
                cleaned = self.url_pattern.sub(" ", cleaned)

            if self.config.remove_emails:
                cleaned = self.email_pattern.sub(" ", cleaned)

            if self.config.handle_emojis:
                cleaned = self.emoji_pattern.sub(" ", cleaned)

            if not self.config.preserve_sentiment_punctuation:
                cleaned = self.non_sentiment_punct.sub(" ", cleaned)

            if self.config.lowercase:
                cleaned = cleaned.lower()

            if self.config.normalize_whitespace:
                cleaned = self.whitespace_pattern.sub(" ", cleaned).strip()

            return cleaned

        except Exception as exc:
            raise PreprocessingError(f"Text preprocessing failed: {str(exc)}") from exc

    def remove_stopwords_preserving_negations(
        self, text: str, custom_stopwords: Set[str] | None = None
    ) -> str:
        """Remove stopwords while strictly preserving any tokens in NEGATION_WHITELIST.

        Args:
            text (str): Input text string.
            custom_stopwords (Optional[Set[str]]): Stopword set to filter against.

        Returns:
            str: Filtered text with negations intact.
        """
        stopwords = custom_stopwords or set()
        # Ensure negation words are NEVER in the stopwords set
        active_stopwords = stopwords - NEGATION_WHITELIST
        words = text.split()
        filtered = [w for w in words if w.lower() not in active_stopwords or w.lower() in NEGATION_WHITELIST]
        return " ".join(filtered)

    def transform_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a list of raw text strings.

        Args:
            texts (List[str]): List of raw texts.

        Returns:
            List[str]: List of cleaned texts.
        """
        return [self.clean_text(t) for t in texts]
