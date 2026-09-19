"""Cap the unbounded reset retry in LIBERO's ControlEnv.

The stock loop retries forever on RandomizationError:

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue
        return ret

A pathological randomisation therefore hangs the whole evaluation (observed:
19 min pinned at one core, no log output, zero I/O, zero GPU -- and with no way
to get a stack inside this container). This converts it into a clean exception
the caller can record and skip.
"""
import pathlib
import shutil

P = pathlib.Path(
    "/root/autodl-tmp/libero-mem/libero/libero/envs/env_wrapper.py")
src = P.read_text(encoding="utf-8")

OLD = '''    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret
'''

NEW = '''    def reset(self):
        # RPBE: stock loop is UNBOUNDED; a pathological randomisation retries
        # forever and hangs the whole eval (seen: 19 min pinned at 1 core, no
        # output, no I/O, no GPU). Cap it so the caller can record + skip.
        import time as _time
        _t0 = _time.time()
        _attempts = 0
        while True:
            try:
                return self.env.reset()
            except RandomizationError:
                _attempts += 1
                if _attempts >= 100 or (_time.time() - _t0) > 120:
                    raise RuntimeError(
                        "env reset failed %d times in %.0fs; giving up"
                        % (_attempts, _time.time() - _t0))
                continue
'''

n = src.count(OLD)
if n != 1:
    raise SystemExit(f"ABORT: found {n} matches of the reset block, expected 1")

shutil.copy2(P, str(P) + ".bak")
P.write_text(src.replace(OLD, NEW), encoding="utf-8")
print("patched:", P)
print("backup :", str(P) + ".bak")
