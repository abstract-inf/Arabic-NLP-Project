# Preprocessed Datasets

This folder contains CSV variants of AAFAQ_Dataset.csv with different preprocessing scopes.

| File | Processing applied | Notes |
| --- | --- | --- |
| pyarabic_aggressive_preprocessed.csv | Tashkeel removed, Tatweel removed, Hamza normalized (tasheel), stopwords removed, punctuation removed, non-letter characters removed, whitespace normalized | Most aggressive cleanup using PyArabic + NLTK |
| regex_aggressive_preprocessed.csv | Tashkeel removed, Tatweel removed, Hamza normalized to ي (U+064A), stopwords removed, punctuation/symbols removed, whitespace normalized | Regex-based aggressive cleanup |
| pyarabic_tashkeel_tatweel_preprocessed.csv | Tashkeel removed, Tatweel removed | No other changes |
| pyarabic_hamza_only_preprocessed.csv | Hamza normalized (tasheel) | Only hamza changes |
| pyarabic_hamza_tashkeel_preprocessed.csv | Tashkeel removed, Hamza normalized (tasheel) | No tatweel removal |
| pyarabic_punctuation_only_preprocessed.csv | Punctuation removed (non-alphanumeric stripped) | Keeps letters, digits, spaces |
