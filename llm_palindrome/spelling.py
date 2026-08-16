"""How a palindrome is written down, as opposed to what it is made of.

The mirror is over letters. Spaces, case, apostrophes and full stops are
invisible to it, which is the licence every catalogued palindrome already
takes: "A man, a plan, a canal: Panama" mirrors as `amanaplanacanalpanama`,
and "Madam, I'm Adam" is stored here as `madaminedenimadam` — an apostrophe
the mirror never sees.

Taking that licence is worth more than it looks. The search can only place
letter strings, so "i am" is the only thing it can build; a reader shown "im"
sees a typo and shown "I'm" sees a contraction. The word is the same word.

Two rules, and both are conservative on purpose:

- Only contractions whose apostrophised form is the *dominant* reading are
  applied. "its" is left alone, because "its" is a word and rewriting it as
  "it's" changes the sentence rather than spelling it. Same for "lets", "were",
  "well", "hell", "shell", "id" and "wed" — all ordinary words.
- A standalone "i" is always capitalised. The search emits lowercase and
  `str.capitalize` lowercases everything after the first letter, so the shipped
  paragraph has been printing "on taxes i moan" with a lowercase pronoun.
"""
from __future__ import annotations

from typing import Sequence

# Spellings where the apostrophe is the only reading. Each key is a letter
# string the vocabulary can actually produce; the value is what a reader sees.
CONTRACTIONS = {
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",          # "ill" is a word — see BARE below
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "youre": "you're",
    "youve": "you've",
    "youll": "you'll",
    "theyre": "they're",
    "theyve": "they've",
    "theyll": "they'll",
    "weve": "we've",
    "wheres": "where's",
    "thats": "that's",
    "whats": "what's",
    "hes": "he's",
    "shes": "she's",
    "theres": "there's",
    "heres": "here's",
    "whos": "who's",
    "aint": "ain't",
}

# Letter strings that are ordinary words in their own right, whatever else
# they could be apostrophised into. Listed rather than omitted so the reason is
# on the page: dropping "ill" from CONTRACTIONS would lose "I'll", and leaving
# it in unconditionally would print "I'll" for "an ill wind".
BARE = frozenset({"its", "lets", "were", "well", "hell", "shell", "id", "wed",
                  "ill"})


def spell_word(word: str, *, contract: bool = True) -> str:
    """One word as it should be printed."""
    if word == "i":
        return "I"
    if contract and word in CONTRACTIONS and word not in BARE:
        return CONTRACTIONS[word]
    return word


def spell(words: Sequence[str], *, contract: bool = True,
          period: bool = True) -> str:
    """A sentence, written the way a reader should see it.

    `str.capitalize` is not enough on its own: it lowercases everything after
    the first character, so any word this function has already spelled with a
    capital — "I", "I'm" — would be undone by it.
    """
    if not words:
        return ""
    out = [spell_word(w, contract=contract) for w in words]
    if out[0][:1].islower():
        out[0] = out[0][0].upper() + out[0][1:]
    text = " ".join(out)
    return text + "." if period else text


def letters(text: str) -> str:
    """What the mirror actually sees. Everything spelling adds is invisible."""
    return "".join(c.lower() for c in text if c.isalpha())
