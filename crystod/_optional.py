"""Guards for the optional dependencies of CrystOD.

PySCF is the only optional dependency: it powers the quantitative engines
(``crystod --diagram/--band/--dos/--visualize --pyscf`` and
``crystod-mol --diagram --pyscf``) and nothing else. Since v0.4.0 a plain
``pip install CrystOD`` does not pull it in; ``pip install "CrystOD[quantum]"``
does.

Every code path that needs PySCF calls :func:`require_pyscf` first, so a
missing installation is reported in one sentence that names the remedy
instead of a ``ModuleNotFoundError`` from deep inside an SCF driver.
"""

from __future__ import annotations

PYSCF_INSTALL_HINT = 'pip install "CrystOD[quantum]"'


def pyscf_available() -> bool:
    """Return ``True`` when PySCF can be imported."""
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return False
    return True


def require_pyscf(feature: str = "this feature") -> None:
    """Raise ``ImportError`` with an installation hint when PySCF is missing.

    Args:
        feature: What the caller is about to do, named the way the user
            invoked it (``"crystod --diagram --pyscf"``, ``"PyscfDiagram"``),
            so the message says which request cannot be served.

    Raises:
        ImportError: PySCF is not installed. The message names the
            ``pip install "CrystOD[quantum]"`` remedy.
    """
    try:
        import pyscf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"PySCF is required for {feature}, but it is not installed. "
            f"Install it with: {PYSCF_INSTALL_HINT} -- PySCF has been an "
            "optional dependency of CrystOD since v0.4.0; the symmetry, "
            "extended-Hueckel, phonon and group-theory features do not "
            "need it."
        ) from exc
