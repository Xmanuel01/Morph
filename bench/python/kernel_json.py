import json


def run() -> int:
    payload = '{"a":1,"b":[1,2,3],"c":"enkai","ok":true}'
    acc = 0
    i = 0
    while i < 2000:
        row = json.loads(payload)
        _text = json.dumps(row, separators=(",", ":"), sort_keys=False)
        acc += 1
        i += 1
    return acc


if __name__ == "__main__":
    print(run())
