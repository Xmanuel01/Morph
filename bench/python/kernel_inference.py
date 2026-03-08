def run() -> bool:
    state = 1337
    acc = 0
    for _ in range(500_000):
        state = (state * 1103515245 + 12345) % 2147483647
        token = state % 32000
        acc += token
    return acc > 0


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
