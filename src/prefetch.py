"""Préchargement par thread : un producteur calcule les éléments d'un itérable (file bornée), l'ordre est conservé."""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterable, Iterator
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
_END = object()
POLL_S = 0.05


class _Failure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


def prefetch(items: Iterable[T], depth: int = 2) -> Iterator[T]:
    """Itère `items` dans un thread avec au plus `depth` éléments d'avance ; depth <= 0 : sans thread.

    Une exception du producteur est relancée côté consommateur ; fermer le générateur arrête le thread."""
    if depth <= 0:
        yield from items
        return
    buffer: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(obj) -> bool:
        while not stop.is_set():
            try:
                buffer.put(obj, timeout=POLL_S)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in items:
                if not put(item):
                    return
            put(_END)
        except BaseException as exc:  # noqa: BLE001 : transmis au consommateur
            put(_Failure(exc))

    thread = threading.Thread(target=produce, name="prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)
