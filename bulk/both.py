from multiprocessing import Process

from .walker import Walker
from .climber import Climber

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
PROJECT = "mutation-sizes"
ITERATIONS = 100


def a(id, iterations: int, group: str):
    c = Climber(id, group, PROJECT)
    for _ in range(iterations):
        c.iter()
    c.run.finish()


def b(id, iterations: int, group: str):
    w = Walker(id, group, PROJECT)
    for _ in range(iterations):
        w.iter()
    w.run.finish()


def double(id: str, index: int):
    a(id, ITERATIONS, ALPHABET[index])
    b(id, ITERATIONS, ALPHABET[index])


if __name__ == "__main__":
    ids = [
        "7e4amaxv",
        "3rfu5et9",
        "2j2ef6g6",
        "vbz4j581",
        "sizbfm8a",
        "3a9uq87i",
        "28dj0wyi",
        "42ivd5gj",
        "3e6cxhk9",
        "3gu1xwsx",
    ]
    threads = []
    for i, id in enumerate(ids):
        threads.append(Process(target=double, args=(id, i)))
    for _, p in enumerate(threads):
        p.start()
    for _, p in enumerate(threads):
        p.join()