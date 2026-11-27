"""Contractions and elisions: an apostrophe costs no letters, so each of these
is a fresh letter string the mirror can pay with ("dont", "oer", "tis").

Only forms whose letters do not already spell a common word are listed:
"we're" would print as "were" and "I'll" as "ill", so they stay out.
"""

# letters -> printed form
SPELL = {
    "dont": "don't", "didnt": "didn't", "doesnt": "doesn't", "isnt": "isn't",
    "wasnt": "wasn't", "werent": "weren't", "arent": "aren't", "hasnt": "hasn't",
    "havent": "haven't", "hadnt": "hadn't", "couldnt": "couldn't", "wouldnt": "wouldn't",
    "shouldnt": "shouldn't", "mustnt": "mustn't", "neednt": "needn't", "aint": "ain't",
    "cant": "can't", "wont": "won't",
    "im": "I'm", "ive": "I've",
    "youre": "you're", "youve": "you've", "youll": "you'll", "youd": "you'd",
    "theyre": "they're", "theyve": "they've", "theyll": "they'll", "theyd": "they'd",
    "weve": "we've", "hed": "he'd", "hes": "he's", "shes": "she's",
    "thats": "that's", "whats": "what's", "theres": "there's", "heres": "here's",
    "wheres": "where's", "whos": "who's", "itll": "it'll", "thatll": "that'll",
    "oclock": "o'clock", "em": "'em",
    # Poetic elisions.
    "tis": "'tis", "twas": "'twas", "oer": "o'er", "eer": "e'er", "neer": "ne'er",
    "neath": "'neath", "gainst": "'gainst", "tween": "'tween",
}

# Forms Brown never tags, read like the word they shorten.
TAG_LIKE = {"neednt": "cant", "twas": "tis", "eer": "ever", "neer": "never",
            "neath": "beneath", "gainst": "against", "tween": "between"}

# printed form, lower case -> letters (how the corpora spell them)
JOIN = {v.lower(): k for k, v in SPELL.items()}
