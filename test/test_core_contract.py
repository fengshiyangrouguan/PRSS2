"""Host-independence red lines and config validation for the rpbe core.

The core modules (compressor / loss / maps / records / config) must never
contain host-specific names — interfaces are keyed by the opaque ``tau`` the
host adapter supplies.  This mirrors the old prss red-line test, pointed at
the new package.
"""

import unittest
from pathlib import Path

from rpbe.config import RPBConfig

SRC = Path(__file__).resolve().parents[1] / "src" / "rpbe"
CORE_FILES = ["compressor.py", "loss.py", "maps.py", "records.py", "config.py"]


class TestHostIndependence(unittest.TestCase):
    def test_core_files_contain_no_host_strings(self):
        for name in CORE_FILES:
            text = (SRC / name).read_text(encoding="utf-8")
            for bad in ("tjo:", "tgp:", "tgw:", "tgn_layer_", "graph_attention"):
                self.assertNotIn(bad, text,
                                 "{} leaks host name {!r}".format(name, bad))


class TestRPBConfigValidation(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(state_dims={"a": 8, "b": 8}, own_dims={"a": 8, "b": 8})
        base.update(kw)
        return RPBConfig(**base)

    def test_valid_config(self):
        cfg = self._cfg()
        self.assertEqual(cfg.alpha("a"), 1.0)

    def test_interface_own_dim_mismatch(self):
        with self.assertRaises(ValueError):
            self._cfg(own_dims={"a": 8})

    def test_nonpositive_widths_rejected(self):
        for field in ("width_D", "m", "d_c", "d_f"):
            with self.assertRaises(ValueError):
                self._cfg(**{field: 0})

    def test_nonpositive_r_tau_rejected(self):
        with self.assertRaises(ValueError):
            self._cfg(state_dims={"a": 0, "b": 8}, own_dims={"a": 8, "b": 8})

    def test_alpha_defaults_and_overrides(self):
        cfg = self._cfg(alphas={"a": 0.5})
        self.assertEqual(cfg.alpha("a"), 0.5)
        self.assertEqual(cfg.alpha("b"), 1.0)


if __name__ == "__main__":
    unittest.main()
