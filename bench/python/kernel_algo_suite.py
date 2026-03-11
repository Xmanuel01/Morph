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


def top_k_sum_repeat(values: list[int], k: int, repeats: int) -> int:
    checksum = 0
    for _ in range(repeats):
        checksum += sum(top_k_ints(values, k))
    return checksum


def split_test_count_repeat(total: int, ratio: float, seed: int, repeats: int) -> int:
    checksum = 0
    for _ in range(repeats):
        checksum += int(split_indices(total, ratio, seed)["test_count"])
    return checksum


def run() -> int:
    top_sum = top_k_sum_repeat([9, 1, 8, 2, 7, 3, 6, 4, 5], 4, 20_000)
    split_sum = split_test_count_repeat(20, 0.2, 42, 20_000)
    if top_sum <= 0:
        return 1
    if split_sum != 80_000:
        return 1
    return 0


if __name__ == "__main__":
    print(run())
