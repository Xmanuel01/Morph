import heapq
from hashlib import sha256


def top_k_ints(values: list[int], k: int) -> list[int]:
    if k <= 0:
        return []
    heap: list[int] = []
    for value in values:
        if len(heap) < k:
            heapq.heappush(heap, value)
            continue
        if value > heap[0]:
            heapq.heapreplace(heap, value)
    return sorted(heap, reverse=True)


def split_indices(total: int, ratio: float, seed: int) -> dict[str, object]:
    ratio = max(0.0, min(1.0, ratio))
    seed_u64 = seed & ((1 << 64) - 1)
    indices = list(range(total))
    indices.sort(
        key=lambda idx: (
            int.from_bytes(
                sha256(seed_u64.to_bytes(8, "little", signed=False) + idx.to_bytes(8, "little")).digest()[:8],
                "little",
            ),
            idx,
        )
    )
    test_count = round(total * ratio)
    train_count = total - test_count
    return {
        "train": indices[:train_count],
        "test": indices[train_count:],
        "test_count": test_count,
    }


def run() -> int:
    for _ in range(20_000):
        top = top_k_ints([9, 1, 8, 2, 7, 3, 6, 4, 5], 4)
        split = split_indices(20, 0.2, 42)
        if top == []:
            return 1
        if split["test_count"] != 4:
            return 1
    return 0


if __name__ == "__main__":
    print(run())
