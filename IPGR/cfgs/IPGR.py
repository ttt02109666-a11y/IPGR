"""
IPGR: Iterative Partial-Guided Refinement
用于自监督点云补全的后处理框架

创新点:
1. 部分点云引导细化 (Partial-Guided Refinement)
2. 多轮迭代细化 (Multi-Round Iterative Refinement)
3. 距离感知自适应 (Distance-Aware Adaptive Refinement)

使用方法:
    from models.IPGR import IPGR

    ipgr = IPGR(base_alpha=0.10, num_iter=4)

    # 推理时
    pred = model(partial)  # 任意点云补全模型的预测
    refined = ipgr(pred, partial)  # 应用 IPGR 后处理
"""

import torch
import torch.nn as nn


class IPGR(nn.Module):
    """
    Iterative Partial-Guided Refinement (IPGR)

    一个通用的点云补全后处理框架，可应用于任何点云补全模型。

    Args:
        base_alpha: 基础细化强度，默认 0.10
        num_iter: 迭代次数，默认 4
        use_distance_aware: 是否使用距离感知自适应，默认 True

    Example:
        >>> ipgr = IPGR(base_alpha=0.10, num_iter=4)
        >>> pred = model(partial)  # [B, N, 3]
        >>> refined = ipgr(pred, partial)  # [B, N, 3]
    """

    def __init__(self, base_alpha=0.10, num_iter=4, use_distance_aware=True):
        super().__init__()
        self.base_alpha = base_alpha
        self.num_iter = num_iter
        self.use_distance_aware = use_distance_aware

    def forward(self, pred, partial):
        """
        对预测点云进行迭代细化

        Args:
            pred: 预测的完整点云 [B, N, 3]
            partial: 输入的部分点云 [B, M, 3]

        Returns:
            refined: 细化后的点云 [B, N, 3]
        """
        if self.use_distance_aware:
            return self.distance_aware_refine(pred, partial)
        else:
            return self.basic_refine(pred, partial)

    def basic_refine(self, pred, partial):
        """
        基础迭代细化 (固定 alpha)

        创新点 1 + 2: 部分点云引导 + 多轮迭代
        """
        refined = pred
        B, N, _ = pred.shape
        batch_idx = torch.arange(B, device=refined.device).unsqueeze(1).expand(-1, N)

        for i in range(self.num_iter):
            # 计算每个预测点到输入点云的距离
            dist = torch.cdist(refined, partial)  # [B, N, M]

            # 找到最近的输入点
            min_idx = dist.argmin(dim=-1)  # [B, N]
            nearest = partial[batch_idx, min_idx]  # [B, N, 3]

            # 向最近点移动
            refined = refined + self.base_alpha * (nearest - refined)

        return refined

    def distance_aware_refine(self, pred, partial):
        """
        距离感知自适应迭代细化

        创新点 1 + 2 + 3: 部分点云引导 + 多轮迭代 + 距离感知

        核心思想:
        - 预测点离输入点近 → alpha 大 (附近有真实几何约束，可信)
        - 预测点离输入点远 → alpha 小 (可能是补全区域，保守)
        """
        refined = pred
        B, N, _ = pred.shape
        batch_idx = torch.arange(B, device=refined.device).unsqueeze(1).expand(-1, N)

        for i in range(self.num_iter):
            # 计算每个预测点到输入点云的距离
            dist = torch.cdist(refined, partial)  # [B, N, M]

            # 找到最近的输入点及其距离
            min_dist, min_idx = dist.min(dim=-1)  # [B, N]
            nearest = partial[batch_idx, min_idx]  # [B, N, 3]

            # 距离归一化 (0-1)
            dist_norm = min_dist / (min_dist.max(dim=-1, keepdim=True)[0] + 1e-6)

            # 距离感知 alpha: 距离近 → alpha 大
            # alpha = base_alpha * (1 + (1 - dist_norm))
            # 即: alpha = base_alpha * (2 - dist_norm)
            # 范围: [base_alpha, 2*base_alpha]
            alpha = self.base_alpha * (2.0 - dist_norm)  # [B, N]

            # 向最近点移动
            refined = refined + alpha.unsqueeze(-1) * (nearest - refined)

        return refined

    def __repr__(self):
        return (f"IPGR(base_alpha={self.base_alpha}, "
                f"num_iter={self.num_iter}, "
                f"use_distance_aware={self.use_distance_aware})")


# ============== 便捷函数 ==============

def ipgr_refine(pred, partial, base_alpha=0.10, num_iter=4):
    """
    IPGR 后处理的函数接口

    Args:
        pred: 预测的完整点云 [B, N, 3]
        partial: 输入的部分点云 [B, M, 3]
        base_alpha: 基础细化强度
        num_iter: 迭代次数

    Returns:
        refined: 细化后的点云 [B, N, 3]
    """
    B, N, _ = pred.shape
    batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand(-1, N)

    refined = pred
    for i in range(num_iter):
        dist = torch.cdist(refined, partial)
        min_dist, min_idx = dist.min(dim=-1)
        nearest = partial[batch_idx, min_idx]

        dist_norm = min_dist / (min_dist.max(dim=-1, keepdim=True)[0] + 1e-6)
        alpha = base_alpha * (2.0 - dist_norm)

        refined = refined + alpha.unsqueeze(-1) * (nearest - refined)

    return refined