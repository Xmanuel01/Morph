def run() -> bool:
    step = 0
    w = 0.1
    lr = 0.0005
    while step < 250_000:
        x = (step % 97) + 1
        y = (x * 3) + 7
        pred = w * x
        err = pred - y
        grad = err * x
        w -= lr * grad
        step += 1
    return -1_000_000 < w < 1_000_000


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
