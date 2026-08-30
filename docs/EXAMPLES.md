# Examples

Generated 25 August 2026 from `runs/punct/`. Every text here is a **valid palindrome** — strip case, spaces and marks and it reads the same both ways.

## What punctuation does to a palindrome

Every row is **one palindrome**. All four versions carry *identical letters in identical order* — only the marks change, so the palindrome is intact in all of them. Scores are 0–3 from four judges, in the order `gpt-oss:120b` / `gpt-oss:20b` / `mistral` / `haiku`, answering "is this real English that makes some kind of sense?"

Rows are sorted by how badly our shipped `present.py` damaged the text, worst first.

### 1. Palindromes people wrote

### `norma is as selfless as i am ron`

| version | text | scores |
|---|---|---|
| **hand** | Norma is as selfless as I am, Ron. | 3 / 3 / 3 / 3 |
| **LLM, after the search** | Norma is as selfless as I am, Ron. | 3 / 3 / 3 / 3 |
| bare — no marks at all | norma is as selfless as i am ron | 2 / 2 / 2 / 3 |
| `present.py`, as shipped | Norma is as. Selfless, as I am. Ron. | 0 / 0 / 2 / 3 |

### `drab as a fool aloof as a bard`

| version | text | scores |
|---|---|---|
| **hand** | Drab as a fool, aloof as a bard. | 3 / 2 / 2 / 2 |
| **LLM, after the search** | Drab as a fool, aloof as a bard. | 3 / 2 / 2 / 3 |
| bare — no marks at all | drab as a fool aloof as a bard | 1 / 2 / 2 / 3 |
| `present.py`, as shipped | Drab, as, a fool aloof. As a bard. | 1 / 0 / 2 / 2 |

### `sir i demand i am a maid named iris`

| version | text | scores |
|---|---|---|
| **hand** | Sir, I demand: I am a maid named Iris. | 3 / 3 / 3 / 3 |
| **LLM, after the search** | Sir, I demand I am a maid named Iris. | 3 / 2 / 3 / 2 |
| bare — no marks at all | sir i demand i am a maid named iris | 2 / 2 / 2 / 3 |
| `present.py`, as shipped | Sir I demand. I am a. Maid, named, iris. | 1 / 0 / 2 / 3 |

### `sums are not set as a test on erasmus`

| version | text | scores |
|---|---|---|
| **hand** | Sums are not set as a test on Erasmus. | 2 / 1 / 3 / 3 |
| **LLM, after the search** | Sums are not set as a test on Erasmus. | 2 / 1 / 3 / 3 |
| bare — no marks at all | sums are not set as a test on erasmus | 2 / 1 / 3 / 2 |
| `present.py`, as shipped | Sums are not. Set, as a test. On, erasmus. | 0 / 1 / 1 / 3 |

### `do nine men interpret nine men i nod`

| version | text | scores |
|---|---|---|
| **hand** | Do nine men interpret? Nine men, I nod. | 2 / 2 / 2 / 3 |
| **LLM, after the search** | Do nine men interpret nine men? I nod. | 2 / 3 / 2 / 3 |
| bare — no marks at all | do nine men interpret nine men i nod | 1 / 0 / 2 / 0 |
| `present.py`, as shipped | Do, nine men interpret. Nine, men I nod. | 0 / 1 / 2 / 3 |

### `campus motto bottoms up mac`

| version | text | scores |
|---|---|---|
| **hand** | Campus motto: bottoms up, Mac! | 3 / 2 / 2 / 0 |
| **LLM, after the search** | Campus motto: bottoms up, mac. | 2 / 2 / 2 / 3 |
| bare — no marks at all | campus motto bottoms up mac | 0 / 0 / 1 / 1 |
| `present.py`, as shipped | Campus, motto, bottoms up mac. | 1 / 0 / 2 / 3 |

### `may a moody baby doom a yam`

| version | text | scores |
|---|---|---|
| **hand** | May a moody baby doom a yam? | 2 / 2 / 2 / 0 |
| **LLM, after the search** | May a moody baby doom a yam. | 1 / 2 / 2 / 3 |
| bare — no marks at all | may a moody baby doom a yam | 1 / 2 / 2 / 3 |
| `present.py`, as shipped | May a moody: baby doom a, yam. | 0 / 0 / 1 / 2 |

### `ten animals i slam in a net`

| version | text | scores |
|---|---|---|
| **hand** | Ten animals I slam in a net. | 3 / 1 / 2 / 3 |
| **LLM, after the search** | Ten animals I slam in a net. | 2 / 1 / 2 / 3 |
| bare — no marks at all | ten animals i slam in a net | 1 / 1 / 2 / 3 |
| `present.py`, as shipped | Ten, animals I slam: in a net. | 1 / 0 / 2 / 2 |

### `no pet so tragic as a cigar to step on`

| version | text | scores |
|---|---|---|
| **hand** | No pet so tragic as a cigar to step on. | 2 / 0 / 2 / 3 |
| **LLM, after the search** | No pet—so tragic as a cigar to step on. | 1 / 0 / 2 / 3 |
| bare — no marks at all | no pet so tragic as a cigar to step on | 1 / 0 / 2 / 3 |
| `present.py`, as shipped | No, pet so tragic: as a cigar, to step on. | 0 / 0 / 2 / 3 |

### `eva can i see bees in a cave`

| version | text | scores |
|---|---|---|
| **hand** | Eva, can I see bees in a cave? | 3 / 2 / 3 / 3 |
| **LLM, after the search** | Eva, can I see bees in a cave? | 3 / 2 / 3 / 3 |
| bare — no marks at all | eva can i see bees in a cave | 3 / 0 / 2 / 3 |
| `present.py`, as shipped | Eva, can, I see bees. In a cave. | 1 / 1 / 2 / 3 |

### `some men interpret nine memos`

| version | text | scores |
|---|---|---|
| **hand** | Some men interpret nine memos. | 3 / 3 / 3 / 2 |
| **LLM, after the search** | Some men interpret nine memos. | 3 / 3 / 3 / 3 |
| bare — no marks at all | some men interpret nine memos | 3 / 3 / 3 / 3 |
| `present.py`, as shipped | Some men interpret. Nine, memos. | 1 / 1 / 2 / 3 |

### `no it can assess an action`

| version | text | scores |
|---|---|---|
| **hand** | No, it can assess an action. | 3 / 3 / 3 / 0 |
| **LLM, after the search** | No, it can assess an action. | 3 / 3 / 3 / 3 |
| bare — no marks at all | no it can assess an action | 2 / 0 / 2 / 3 |
| `present.py`, as shipped | No it can. Assess an action. | 1 / 2 / 2 / 2 |

### `a man a plan a canal panama`

| version | text | scores |
|---|---|---|
| **hand** | A man, a plan, a canal: Panama. | 3 / 3 / 3 / 3 |
| **LLM, after the search** | A man, a plan, a canal, Panama! | 3 / 3 / 3 / 3 |
| bare — no marks at all | a man a plan a canal panama | 3 / 1 / 3 / 3 |
| `present.py`, as shipped | A, man a plan. A canal panama. | 1 / 0 / 2 / 2 |

### `cigar toss it in a can it is so tragic`

| version | text | scores |
|---|---|---|
| **hand** | Cigar? Toss it in a can. It is so tragic. | 2 / 2 / 2 / 3 |
| **LLM, after the search** | Cigar toss it in a can. It is so tragic. | 1 / 2 / 2 / 3 |
| bare — no marks at all | cigar toss it in a can it is so tragic | 1 / 1 / 2 / 0 |
| `present.py`, as shipped | Cigar toss it. In a can. It is so. Tragic. | 0 / 1 / 2 / 2 |

### `straw no too stupid a fad i put soot on warts`

| version | text | scores |
|---|---|---|
| **hand** | Straw? No, too stupid a fad. I put soot on warts. | 2 / 1 / 2 / 1 |
| **LLM, after the search** | Straw? No—too stupid a fad. I put soot on warts. | 2 / 2 / 2 / 2 |
| bare — no marks at all | straw no too stupid a fad i put soot on warts | 1 / 0 / 2 / 3 |
| `present.py`, as shipped | Straw no too: stupid a fad, I put soot. On, warts. | 0 / 0 / 1 / 2 |

### `doc note i dissent a fast never prevents a fatness i diet on cod`

| version | text | scores |
|---|---|---|
| **hand** | Doc, note: I dissent. A fast never prevents a fatness. I diet on cod. | 2 / 2 / 2 / 3 |
| **LLM, after the search** | Doc note: I dissent. A fast never prevents a fatness. I diet on cod. | 3 / 2 / 2 / 3 |
| bare — no marks at all | doc note i dissent a fast never prevents a fatness i diet on cod | 1 / 0 / 1 / 2 |
| `present.py`, as shipped | Doc note I. Dissent a fast: never prevents a. Fatness, I, diet on cod. | 1 / 0 / 2 / 1 |

### `rats live on no evil star`

| version | text | scores |
|---|---|---|
| **hand** | Rats live on no evil star. | 3 / 2 / 2 / 0 |
| **LLM, after the search** | Rats live on no evil star. | 3 / 2 / 2 / 3 |
| bare — no marks at all | rats live on no evil star | 2 / 2 / 2 / 3 |
| `present.py`, as shipped | Rats live on. No evil star. | 2 / 2 / 2 / 3 |

### `are we not drawn onward we few drawn onward to new era`

| version | text | scores |
|---|---|---|
| **hand** | Are we not drawn onward, we few, drawn onward to new era? | 2 / 1 / 3 / 2 |
| **LLM, after the search** | Are we not drawn onward? We few, drawn onward to new era. | 2 / 2 / 3 / 3 |
| bare — no marks at all | are we not drawn onward we few drawn onward to new era | 1 / 1 / 2 / 3 |
| `present.py`, as shipped | Are, we, not drawn onward. We few drawn, onward to new: era. | 1 / 0 / 2 / 0 |

### `a tin mug for a jar of gum nit a`

| version | text | scores |
|---|---|---|
| **hand** | A tin mug for a jar of gum, nit a. | 1 / 0 / 2 / 1 |
| **LLM, after the search** | A tin mug for a jar of gum—nit a. | 0 / 0 / 2 / 3 |
| bare — no marks at all | a tin mug for a jar of gum nit a | 0 / 0 / 1 / 2 |
| `present.py`, as shipped | A, tin, mug for a: jar of gum, nit, a. | 0 / 0 / 2 / 2 |

### `murder for a jar of red rum`

| version | text | scores |
|---|---|---|
| **hand** | Murder for a jar of red rum. | 2 / 2 / 2 / 3 |
| **LLM, after the search** | Murder for a jar of red rum. | 2 / 2 / 2 / 3 |
| bare — no marks at all | murder for a jar of red rum | 2 / 2 / 2 / 3 |
| `present.py`, as shipped | Murder, for a jar: of red rum. | 1 / 1 / 2 / 3 |

### `golf no sir prefer prison flog`

| version | text | scores |
|---|---|---|
| **hand** | Golf? No, sir. Prefer prison. Flog! | 2 / 2 / 2 / 3 |
| **LLM, after the search** | Golf? No, sir—prefer prison, flog. | 1 / 2 / 2 / 0 |
| bare — no marks at all | golf no sir prefer prison flog | 0 / 0 / 1 / 0 |
| `present.py`, as shipped | Golf no sir: prefer prison flog. | 1 / 0 / 2 / 3 |

### `go deliver a dare vile dog`

| version | text | scores |
|---|---|---|
| **hand** | Go deliver a dare, vile dog! | 2 / 1 / 2 / 3 |
| **LLM, after the search** | Go deliver a dare, vile dog. | 2 / 1 / 2 / 0 |
| bare — no marks at all | go deliver a dare vile dog | 0 / 0 / 2 / 0 |
| `present.py`, as shipped | Go deliver a. Dare vile dog. | 1 / 0 / 2 / 1 |

### `no sir away a papaya war is on`

| version | text | scores |
|---|---|---|
| **hand** | No, sir! Away! A papaya war is on! | 2 / 2 / 2 / 3 |
| **LLM, after the search** | No, sir—away! A papaya war is on. | 2 / 2 / 2 / 3 |
| bare — no marks at all | no sir away a papaya war is on | 0 / 1 / 1 / 2 |
| `present.py`, as shipped | No, sir away a: papaya, war is on. | 1 / 0 / 1 / 2 |

### `a dog a plan a canal pagoda`

| version | text | scores |
|---|---|---|
| **hand** | A dog, a plan, a canal: Pagoda. | 1 / 1 / 3 / 2 |
| **LLM, after the search** | A dog, a plan, a canal, pagoda. | 1 / 0 / 2 / 3 |
| bare — no marks at all | a dog a plan a canal pagoda | 1 / 1 / 2 / 3 |
| `present.py`, as shipped | A, dog a plan. A canal pagoda. | 0 / 0 / 1 / 3 |

### `satan oscillate my metallic sonatas`

| version | text | scores |
|---|---|---|
| **hand** | Satan, oscillate my metallic sonatas! | 2 / 2 / 2 / 3 |
| **LLM, after the search** | Satan oscillate my metallic sonatas. | 2 / 2 / 2 / 2 |
| bare — no marks at all | satan oscillate my metallic sonatas | 1 / 1 / 1 / 1 |
| `present.py`, as shipped | Satan, oscillate, my metallic sonatas. | 1 / 1 / 2 / 3 |

### `stressed was i ere i saw desserts`

| version | text | scores |
|---|---|---|
| **hand** | Stressed was I ere I saw desserts. | 2 / 2 / 2 / 2 |
| **LLM, after the search** | Stressed was I ere I saw desserts. | 2 / 2 / 2 / 3 |
| bare — no marks at all | stressed was i ere i saw desserts | 2 / 2 / 2 / 3 |
| `present.py`, as shipped | Stressed was I: ere, I saw desserts. | 2 / 2 / 2 / 3 |

### 2. What our search actually finds

Same treatment, on text this project generated rather than text people wrote. `k` is the number of chunks nested — `k=1` is a single find, higher `k` glues more chunks around it. Every one is a valid palindrome.

### k=1 — 30 letters

| version | text |
|---|---|
| bare | `part sir or recipe epic error is trap` |
| **LLM, after** | Part, sir, or recipe epic error is trap. |
| `present.py` | Part, sir, or recipe epic: error is trap. |

### k=1 — 30 letters

| version | text |
|---|---|
| bare | `draw error recipe epic error reward` |
| **LLM, after** | Draw error. Recipe epic error reward. |
| `present.py` | Draw error recipe: epic error reward. |

### k=1 — 30 letters

| version | text |
|---|---|
| bare | `mall is mar gets at taste grams ill am` |
| **LLM, after** | Mall is mar; gets at taste, grams ill am. |
| `present.py` | Mall is mar. Gets at taste: grams ill am. |

### k=2 — 58 letters

| version | text |
|---|---|
| bare | `lie net at seat set a to of met system met system foot a test a estate neil` |
| **LLM, after** | Lie net at seat set a to of met system. Met system foot a test a estate neil. |
| `present.py` | Lie net at, seat set a. To of met. System met system. Foot a test. A estate neil. |

### k=2 — 60 letters

| version | text |
|---|---|
| bare | `a pack cent filled levels draw award draw awards level del lift neck cap a` |
| **LLM, after** | A pack cent filled levels; draw award, draw awards; level del lift neck cap a. |
| `present.py` | A pack cent. Filled levels draw: award draw awards. Level del lift. Neck cap a. |

### k=4 — 116 letters

| version | text |
|---|---|
| bare | `dam rat snow order noon a estimates a level named art a flow path submit tim bush tap wolf a trade man level a set am it sea noon red row on star mad` |
| **LLM, after** | Dam rat snow order noon. A estimates a level named art. A flow path submit tim bush tap wolf. A trade man level a set am it sea noon red row on star mad. |
| `present.py` | Dam, rat snow order. Noon a estimates. A level named. Art a flow. Path submit tim. Bush tap wolf. A trade man. Level a set. Am it sea. Noon red row. On star mad. |

### k=8 — 234 letters

| version | text |
|---|---|
| bare | `ill is mar gets at bus sir or recipe no jane estimates net pet snow order no i tom estimates tap sir or recipe no cast estimates i pool none dragon no garden on loop i set am it sets a con epic error is pat set am its emotion red row on step ten set am it seen a jon epic error is sub taste grams ill i` |
| **LLM, after** | Ill is mar gets at bus, sir, or recipe? No, Jane estimates net pet snow order. No, I Tom estimates tap, sir, or recipe. No, cast estimates I pool none dragon. No garden on loop. I set am it sets a con epic error. Is pat set am its emotion red row on step ten? Set am it seen a Jon epic error is sub taste grams. Ill I. |
| `present.py` | Ill is mar. Gets at bus: sir or recipe, no jane estimates. Net pet snow. Order no I, tom estimates tap. Sir or recipe, no cast estimates. I pool none. Dragon, no garden on. Loop I set. Am it sets. A con epic, error is pat. Set am its. Emotion, red row on. Step ten set. Am it seen. A jon epic, error is sub. Taste, grams ill I. |


### 3. The 54-letter construction that beats the human record on length

Two of our chunks nested. **54 letters**, valid — longer than the 51-letter human best, which has stood since 1984.

| | text | letters |
|---|---|---|
| human record | Doc, note: I dissent. A fast never prevents a fatness. I diet on cod. | 51 |
| **ours, nested** | Non academia, Reno, Sir parasites set I Sara prisoner, aimed a canon. | **54** |
| chunk A alone | non academia … aimed a canon | 22 |
| chunk B alone | reno sir parasites … set i sara prisoner | 32 |

It is longer. It reads worse than either half it was built from — blind judging prefers a single chunk to a nest **14/14**, and the collapse is already total at one seam.
