# Human scene equation frames

This experiment uses three complete, human-authored ordinary-order scene
frames. Each frame is realized from semantic slots (agent, action, object,
place), and the constructor records the live unmatched character debt after
every slot append. It therefore exposes where a readable scene fails to close,
instead of treating readability as a late filter.

Primary rendering (145 letters):

> The patient courier delivered the sealed letter to the quiet office. The waiting clerk opened the letter beside the window. The grateful clerk thanked the courier before noon.

The held-out repair changes only the response-frame place slot (142 letters):

> The patient courier delivered the sealed letter to the quiet office. The waiting clerk opened the letter at the old desk. The grateful clerk thanked the courier before noon.

Both renderings are intact ordinary prose, with provenance and a live slot
trace in `runs/human-scene-equation-frames-20260916.json`. Independent tape
audits agree: both are non-palindromes (`audit_a=false`, `audit_b=false`). No
candidate is promoted and no reader claim is made. The next repair is to
author a second response frame whose semantic place phrase is selected against
the outstanding outer-character debt, then rerun the complete-frame equation
and the blinded intact-prose reader gate.
