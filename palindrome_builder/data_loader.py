def load_word_list(source="nltk"):
    """
    Loads a list of English words.
    Can extend to external corpora.
    """
    if source == "nltk":
        try:
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
        except Exception:
            # Fallback: minimal set
            return set(["level", "radar", "noon", "civic", "madam", "refer", "kayak", "rotor", "stats", "mom", "dad"])
    elif source == "wordfreq":
        try:
            from wordfreq import top_n_list
            return set(top_n_list("en", 50000))
        except Exception:
            return set()
    else:
        return set()
