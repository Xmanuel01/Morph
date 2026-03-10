import hashlib


def run() -> int:
    payload = b"enkai benchmark payload"
    digests = bytearray()
    for _ in range(80_000):
        digests.extend(hashlib.sha256(payload).digest())
    return len(digests)


if __name__ == "__main__":
    print(run())
