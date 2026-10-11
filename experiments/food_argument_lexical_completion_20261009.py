"""Predeclared 006 extension: recipient-food vs affected-human argument slots.

005 remains reproducible with its default grammar. This run uses the same
CPU/work/time/path bounds and new seed; all prior outputs stay untouched.
"""
from pathlib import Path
from experiments import bidirectional_lexical_completion_20261009 as base

def run():
    original=base.frames
    previous_out,previous_seed=base.OUT,base.SEED
    try:
        base.frames=lambda:original(include_food=True)
        base.OUT=Path('research/block-seams/food-argument-lexical-006')
        base.SEED='food-argument-lexical-006'
        return base.run()
    finally:
        base.frames=original
        base.OUT,base.SEED=previous_out,previous_seed

if __name__=='__main__':run()
