"""Import shim for the vendored TGB modules.

The vendored files keep their upstream repo-root import layout (``from modules.x
import ...``), which is only resolvable when the TGB repository root is on
``sys.path``.  This shim registers an alias package named ``modules`` whose path
points at this directory, so the files stay byte-identical to upstream.
"""

import sys
import types as _types

_pkg = _types.ModuleType("modules")
_pkg.__path__ = [__path__[0]]  # noqa: F821 (module __path__ defined by importlib)
sys.modules.setdefault("modules", _pkg)
