"""Development-only strict closure gate for the existing search APIs."""
from .admission import mechanical_admission_checks, normalize_letters


def strict_closure_gate(min_letters, max_letters, *, additional_gate=None):
    if min_letters < 1 or max_letters < min_letters:
        raise ValueError('invalid length band')

    def allow_closed(left, right):
        text = ' '.join((*left, *right)).capitalize() + '.'
        try:
            tape = normalize_letters(text)
        except ValueError:
            return False
        if not tape or tape != tape[::-1]:
            return False
        checks = mechanical_admission_checks(text, min_letters=min_letters, max_letters=max_letters)
        return all(checks.values()) and (additional_gate is None or additional_gate(left, right))
    return allow_closed


def best_strict_search(search, tries, scorer, *, min_letters, max_letters, additional_gate=None, **settings):
    """Keep the search's score ordering but only accept strict-eligible closures.

    The caller owns the resource budget and scorer; no inference or training.
    This adapter is for outside-in and empty-center center-out search.
    """
    if settings.get('center', ''):
        raise ValueError('strict adapter requires an empty center')
    if 'allow_closed' in settings:
        raise ValueError('use additional_gate rather than replacing the hard gate')
    result = search(tries, scorer, min_letters=min_letters,
                    allow_closed=strict_closure_gate(min_letters, max_letters,
                                                      additional_gate=additional_gate), **settings)
    if result and not strict_closure_gate(min_letters, max_letters)(tuple(result), ()):
        raise AssertionError('search returned an ineligible closure')
    return result
