import json

def run() -> int:
    acc = 1
    for i in range(1, 500_000):
        acc += ((i * 3) - (i // 2))
    if acc == -1:
        print(acc)
    return acc


if __name__ == "__main__":
    print(run())
