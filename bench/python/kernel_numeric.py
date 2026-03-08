import json

def run() -> int:
    acc = 0
    for i in range(500_000):
        acc += ((i * 31) // 7) - ((i * 3) // 11)
    if acc == -1:
        print(acc)
    return acc


if __name__ == "__main__":
    print(run())
