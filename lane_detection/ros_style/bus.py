"""The notice boards: publish / subscribe, with 'drop the oldest' queues."""
import queue
import threading

from collections import defaultdict



class MessageBus:
    def __init__(self):
        self._subs = defaultdict(list)
        self._counts = defaultdict(int)
        self._lock = threading.Lock()
        self.shutdown = threading.Event()

    def subscribe(self, topic: str, queue_size: int = 1) -> queue.Queue:
        q = queue.Queue(maxsize=queue_size)
        with self._lock:
            self._subs[topic].append(q)
        return q

    def publish(self, topic: str, msg) -> None:
        with self._lock:
            subscribers = list(self._subs[topic])
            self._counts[topic] += 1
        for q in subscribers:
            while True:
                try:
                    q.put_nowait(msg)
                    break
                except queue.Full:                   # board full: throw away the oldest
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass

    @property
    def counts(self) -> dict:
        with self._lock:
            return dict(self._counts)
