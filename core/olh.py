"""
Optimized Local Hashing (OLH) Protocol
========================================
Local Hashing protocol [Wang et al., 2017], which hashes the input into a
smaller domain of size g = e^ε + 1 before applying GRR.

References:
    [1] Wang et al. (2017) "Locally differentially private protocols for frequency
        estimation" (USENIX Security).
    [2] Bassily and Smith "Local, private, efficient protocols for succinct
        histograms" (STOC).
"""

import numpy as np
from sys import maxsize
import xxhash

from .grr import GRR_Client


from ._common import matrix_inversion

import secrets  # 新增导入
def LH_Client(input_data: int, d: int, epsilon: float, optimal: bool = True) -> tuple:
    """
    OLH client-side perturbation for a single user value.

    Each user:
      1. Picks a random hash function h: [d] → [g] (identified by a seed).
      2. Computes y = h(input_data) ∈ [0, g-1].
      3. Applies GRR on y within the reduced domain [g].

    With optimal=True (OLH): g = round(e^ε) + 1.
    With optimal=False (BLH): g = 2.

    Args:
        input_data: The user's true value in [0, d-1].
        d: Domain size.
        epsilon: Privacy budget (ε > 0).
        optimal: If True, use Optimized LH (OLH).

    Returns:
        Tuple of (sanitized_value, rnd_seed) where sanitized_value ∈ [0, g-1].

    Raises:
        ValueError: If input_data is out of range, d < 2, or epsilon <= 0.

    Example:
        >>> report = LH_Client(input_data=3, d=100, epsilon=2.0)
    """
    if input_data < 0 or input_data >= d:
        raise ValueError(f"input_data must be in [0, d-1], got {input_data}.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    if epsilon > 0:
        g = int(round(np.exp(epsilon))) + 1 if optimal else 2

        # 使用 secrets 生成更安全的随机种子（替代 np.random.randint）
        rnd_seed = secrets.randbits(64)  # 64位随机数
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)
        sanitized_value = GRR_Client(hashed_input_data, g, epsilon)

    return (sanitized_value, rnd_seed)


def LH_Aggregator_MI(reports: list, d: int, epsilon: float, optimal: bool = True) -> np.ndarray:
    """
    OLH server-side aggregator using Matrix Inversion (MI).

    For each report (y, seed), counts for value v are incremented whenever
    hash_seed(v) == y. MI is then applied using OLH's p and q parameters.

    Args:
        reports: List of (sanitized_value, rnd_seed) tuples from LH_Client.
        d: Domain size.
        epsilon: Privacy budget.
        optimal: If True, use OLH parameters.

    Returns:
        Array of estimated frequencies (rounded, non-negative).

    Raises:
        ValueError: If reports is empty, d < 2, or epsilon <= 0.

    Example:
        >>> estimates = LH_Aggregator_MI(reports=tuples, d=100, epsilon=2.0)
    """
    if len(reports) == 0:
        raise ValueError("reports list is empty.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    n = len(reports)
    g = int(round(np.exp(epsilon))) + 1 if optimal else 2

    # 预计算所有报告对每个值的哈希匹配
    # 使用列表推导和向量化操作
    count_report = np.zeros(d)

    # 批量处理：预计算所有seed和sanitized_value
    seeds = np.array([seed for _, seed in reports], dtype=np.int64)
    sanitized = np.array([val for val, _ in reports], dtype=np.int32)

    # 对于每个值v，检查哪些报告匹配
    # 注意：由于xxhash不能向量化，仍需循环但可以优化
    for v in range(d):
        # 预计算v的哈希值（对于相同seed可缓存）
        v_str = str(v)
        matches = 0
        for i, (s_val, seed) in enumerate(reports):
            # 可添加LRU缓存
            h = xxhash.xxh32(v_str, seed=seed).intdigest() % g
            if h == s_val:
                matches += 1
        count_report[v] = matches

    p = np.exp(epsilon) / (np.exp(epsilon) + g - 1)
    q = 1.0 / g
    return matrix_inversion(count_report, n, p, q)

from typing import List
class HashFunctionPool:
    """哈希函数池，避免重复生成随机种子"""

    def __init__(self, pool_size: int = 1000):
        self.pool_size = pool_size
        self._seeds = None
        self._current_idx = 0
        self._refresh()

    def _refresh(self):
        self._seeds = [secrets.randbits(64) for _ in range(self.pool_size)]
        self._current_idx = 0

    def get_seed(self) -> int:
        seed = self._seeds[self._current_idx]
        self._current_idx = (self._current_idx + 1) % self.pool_size
        if self._current_idx == 0:
            self._refresh()
        return seed

    def get_seeds(self, count: int) -> List[int]:
        """批量获取多个种子"""
        seeds = []
        for _ in range(count):
            seeds.append(self.get_seed())
        return seeds


# 全局哈希函数池实例
_HASH_POOL = HashFunctionPool()


def LH_Client_Optimized(input_data: int, d: int, epsilon: float, optimal: bool = True) -> tuple:
    """使用哈希函数池的优化版本"""
    # ... 参数验证 ...
    g = int(round(np.exp(epsilon))) + 1 if optimal else 2
    rnd_seed = _HASH_POOL.get_seed()
    hashed = xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g
    sanitized = GRR_Client(hashed, g, epsilon)
    return (sanitized, rnd_seed)