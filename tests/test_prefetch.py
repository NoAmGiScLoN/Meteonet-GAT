"""Préchargement par thread : ordre conservé, erreurs propagées, arrêt propre."""
import threading
import time

import pytest

from prefetch import prefetch


@pytest.mark.unit
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_prefetch_preserves_order(depth):
    assert list(prefetch(iter(range(20)), depth)) == list(range(20))


@pytest.mark.unit
def test_prefetch_propagates_producer_error():
    def gen():
        yield 1
        raise KeyError("boom")
    it = prefetch(gen(), 2)
    assert next(it) == 1
    with pytest.raises(KeyError):
        next(it)


@pytest.mark.unit
def test_prefetch_close_stops_thread_and_queue_is_bounded():
    produced = []

    def gen():
        for i in range(1000):
            produced.append(i)
            yield i
    it = prefetch(gen(), 2)
    assert next(it) == 0
    time.sleep(0.1)
    assert len(produced) <= 5  # file bornée : le producteur attend
    it.close()
    time.sleep(0.2)
    assert not any(t.name == "prefetch" and t.is_alive() for t in threading.enumerate())
