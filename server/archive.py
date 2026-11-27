"""Archive mode: serve the retired v1-v4 generators exactly as they ran live.

Private replay only. Set PALINDROME_ARCHIVE=1 on a localhost-bound instance to
lift the retirement gates in server/app.py, server/v2.py and server/v3.py.
Unset (the default and production), every gate stays in force.
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("PALINDROME_ARCHIVE") == "1"
