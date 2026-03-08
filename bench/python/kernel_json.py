import json


def run() -> int:
    payload = '{"a":1,"b":[1,2,3],"c":"enkai","ok":true}'
    acc = 0
    for _ in range(120_000):
        row = json.loads(payload)
        text = json.dumps(row, separators=(",", ":"), sort_keys=False)
        acc += len(text)
    return acc


if __name__ == "__main__":
    print(run())
