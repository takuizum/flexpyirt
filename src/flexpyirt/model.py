"""Model specification (Spec layer, DESIGN.md §2 ①).

M0 placeholder: only the type exists. Constructors (``uniform``, ``from_items``) and builder
methods (``fix``, ``prior``, ``with_groups`` ...) are added from M1 on (API.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Model"]


@dataclass(frozen=True)
class Model:
    """Immutable specification of an IRT model.

    Builder methods return a new ``Model``; an existing instance is never mutated.
    """
