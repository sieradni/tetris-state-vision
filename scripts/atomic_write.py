"""Atomic replacement with a finite retry for Windows sharing/access races."""
import os
import time


def replace_retry(source, target):
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            # At most150ms total sleep; persistent errors remain fail-closed.
            time.sleep(.01*(attempt+1))
