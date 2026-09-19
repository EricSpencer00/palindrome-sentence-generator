# Synchronous clause product — invalid implementation record

Commit `15401d21` is withdrawn as experiment evidence. Its gate compared only
the first character of each left word with the last character of each paired
right word; it did not consume character streams across word boundaries.
Therefore its zero-candidate result is not a valid discriminator. The added
regression test demonstrates the false-positive shape (`ab`/`ba`). A future
run must use a character trie or stream cursor that preserves grammar state
while consuming every mirrored character.
