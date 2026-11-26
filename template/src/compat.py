"""Python 3.14+ compatibility shim for hydra-core 1.3.x. Delete once hydra-core >= 1.4 is stable.

Python 3.14+ argparse validates help text in `add_argument` and requires a `str`.
Hydra 1.3.x passes a lazy, non-string help object for `--shell-completion`, so
`@hydra.main` fails with "ValueError: badly formed help string". This shim converts
non-string help to `str` before the check; string help is untouched.
"""

from __future__ import annotations

import argparse

if hasattr(argparse.ArgumentParser, "_check_help"):  # Python >= 3.14
    _check_help = argparse.ArgumentParser._check_help

    def _patched_check_help(self: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        _check_help(self, action)

    argparse.ArgumentParser._check_help = _patched_check_help  # type: ignore[method-assign]
