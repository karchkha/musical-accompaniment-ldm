#!/bin/bash
# Patch beat_this for Python 3.9 compatibility.
# beat_this uses X | Y type union syntax (Python 3.10+).
# Adding `from __future__ import annotations` makes it work on 3.9.

BEAT_THIS=$(python -c "import beat_this; import os; print(os.path.dirname(beat_this.__file__))")

for f in inference.py model/loss.py model/postprocessor.py model/beat_tracker.py; do
    if ! grep -q "from __future__ import annotations" "$BEAT_THIS/$f"; then
        sed -i '1s/^/from __future__ import annotations\n/' "$BEAT_THIS/$f"
        echo "Patched: $f"
    else
        echo "Already patched: $f"
    fi
done

echo "Done."
