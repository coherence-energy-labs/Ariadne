"""A test that stubs one import must not evict every module imported alongside it.

``unittest.mock.patch.dict(sys.modules, {...})`` restores sys.modules on exit by
CLEARING it and re-inserting its snapshot, so every module first imported inside the
block is evicted. When that module is astropy, the next import re-runs astropy's
``_init_log`` on the process-global "astropy" logger, whose warnings hook was installed
under pytest's per-test ``catch_warnings`` and has since been restored away:
``LoggingError: Cannot disable warnings logging``. Which later test fails depends on
which xdist worker ran what first, so the failures moved whenever a test module was
added (the 3.10-3.13 CI failures on the ONE Frontier PR) -- but the defect is
reproducible serially on the unmodified base with two tests.
"""

import re
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
EVICTING = re.compile(r"patch\.dict\(\s*(['\"]sys\.modules['\"]|sys\.modules)")


def test_no_test_patches_sys_modules_wholesale():
    offenders = [
        f"{path.name}:{text[: m.start()].count(chr(10)) + 1}"
        for path in sorted(TESTS.glob("*.py"))
        if path.name != Path(__file__).name
        for text in [path.read_text(encoding="utf-8")]
        for m in EVICTING.finditer(text)
    ]
    assert offenders == [], (
        "use pytest.MonkeyPatch.context() + setitem(sys.modules, name, stub), which "
        f"restores only that key: {offenders}"
    )


PROBE = """
import sys, warnings
from unittest.mock import patch
{stub}
warnings.simplefilter("default")
with warnings.catch_warnings():          # what pytest wraps around every test
    with {context}:
        {install}
        import astropy.units                # first import happens inside the stub block
with warnings.catch_warnings():
    import astropy.wcs                      # a later test
print("astropy re-import ok")
"""


def _probe(context, install):
    # numpy/pytest are imported at collection in the real suite, before any stub block;
    # astropy is imported lazily inside a test, which is what gets evicted.
    code = PROBE.format(stub="import numpy, pytest", context=context, install=install)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_mechanism_patch_dict_eviction_breaks_astropy_and_setitem_does_not():
    evicting = _probe('patch.dict("sys.modules", {"astroquery.gaia": None})', "pass")
    scoped = _probe(
        "__import__('pytest').MonkeyPatch.context() as mp",
        "mp.setitem(sys.modules, 'astroquery.gaia', None)",
    )
    assert evicting.returncode != 0 and "LoggingError" in evicting.stderr, evicting.stderr[-500:]
    assert scoped.returncode == 0, scoped.stderr[-500:]
