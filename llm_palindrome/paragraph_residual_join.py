"""Exact joining of unequal-length clause halves via palindromic residuals."""
from collections import defaultdict
import time

class JoinBudgetExceeded(RuntimeError):
    pass

def exact_residual_pairs(left_tapes,right_tapes,*,max_work=20000000,deadline=None):
    """Return occurrence-index pairs where L+R equals its full reversal.

    Half boundaries need not coincide with the character midpoint. All keys
    are full normalized character tapes; caller retains word boundaries and
    grammar/source metadata. Duplicate tape occurrences remain distinct.
    """
    reverse_index=defaultdict(list);long_index=defaultdict(list);work=0
    def check():
        nonlocal work
        work+=1
        if work>max_work or (deadline is not None and time.monotonic()>=deadline):
            raise JoinBudgetExceeded('exact residual join budget exceeded')
    for j,right in enumerate(right_tapes):
        reverse=right[::-1];reverse_index[reverse].append(j)
        for cut in range(len(reverse)+1):
            check();residual=reverse[cut:]
            if residual==residual[::-1]:long_index[reverse[:cut]].append(j)
    pairs=[]
    for i,left in enumerate(left_tapes):
        # |R|>=|L|: reversed R starts with L, leaving a palindrome.
        for j in long_index.get(left,()):check();pairs.append((i,j))
        # |L|>|R|: L starts with reversed R, leaving a palindrome.
        for cut in range(len(left)):
            check();residual=left[cut:]
            if residual==residual[::-1]:
                for j in reverse_index.get(left[:cut],()):check();pairs.append((i,j))
    return pairs,{'work':work,'left_occurrences':len(left_tapes),'right_occurrences':len(right_tapes),'pairs':len(pairs),'complete':True}
