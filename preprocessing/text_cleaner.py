import re 
import emoji

def clean_text(text: str) -> str:
    """
    Cleans social-media style text while preserving sentiment signals."""

    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" "," "))

    # lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https|S+", " " ,text, flags = re.MULTILINE)

    # Normalize whitespace 
    text = re.sub(r"\s+"," ", text).strip()

    # Keep words, underscores (from emojis) and spaces
    text = re.sub(r"[^a-zA-Z0-9_]", " ", text)

    return text
    