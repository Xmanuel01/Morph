import hashlib


def run() -> int:
    payload = b"enkai benchmark payload"
    acc = 0
    for _ in range(80_000):
        digest = hashlib.sha256(payload).digest()
        acc += len(digest)
    return acc


if __name__ == "__main__":
    print(run())
