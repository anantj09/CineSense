from src.components.text_cleaner import (
    TextCleaner
)

cleaner = TextCleaner()

sample_text = "Amazing MOVIE!!!"

cleaned = cleaner.clean_text(
    sample_text
)

assert isinstance(cleaned, str)

print("Preprocessing test passed")