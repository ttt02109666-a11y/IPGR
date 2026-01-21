"""
IPGR 测试脚本

使用方法:
    python test_ipgr.py --config cfgs/xxx.yaml --ckpt path/to/ckpt.pth

可选参数:
    --base_alpha: 基础细化强度 (默认 0.10)
    --num_iter: 迭代次数 (默认 4)
    --no_ipgr: 不使用 IPGR 后处理
"""

import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.getcwd())

from tools import builder
from utils.config import cfg_from_yaml_file
from easydict import EasyDict
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class IPGR(nn.Module):
    """Iterative Partial-Guided Refinement"""

    def __init__(self, base_alpha=0.10, num_iter=4):
        super().__init__()
        self.base_alpha = base_alpha
        self.num_iter = num_iter

    def forward(self, pred, partial):
        refined = pred
        B, N, _ = pred.shape
        batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand(-1, N)

        for i in range(self.num_iter):
            dist = torch.cdist(refined, partial)
            min_dist, min_idx = dist.min(dim=-1)
            nearest = partial[batch_idx, min_idx]

            # 距离感知 alpha
            dist_norm = min_dist / (min_dist.max(dim=-1, keepdim=True)[0] + 1e-6)
            alpha = self.base_alpha * (2.0 - dist_norm)

            refined = refined + alpha.unsqueeze(-1) * (nearest - refined)

        return refined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, required=True, help='模型权重路径')
    parser.add_argument('--base_alpha', type=float, default=0.10, help='IPGR 基础 alpha')
    parser.add_argument('--num_iter', type=int, default=4, help='IPGR 迭代次数')
    parser.add_argument('--no_ipgr', action='store_true', help='不使用 IPGR')
    args = parser.parse_args()

    print('=' * 70)
    print('IPGR: Iterative Partial-Guided Refinement')
    print('=' * 70)
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.ckpt}')
    if not args.no_ipgr:
        print(f'IPGR: base_alpha={args.base_alpha}, num_iter={args.num_iter}')
    else:
        print('IPGR: Disabled')
    print('=' * 70)

    # 加载配置
    config = cfg_from_yaml_file(args.config)
    config = EasyDict(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    class Args:
        distributed = False
        num_workers = 4

    _, test_dataloader = builder.dataset_builder(Args(), config.dataset.test)
    print(f'Test samples: {len(test_dataloader)}')

    # 构建模型
    base_model = builder.model_builder(config.model)

    # 加载权重
    state_dict = torch.load(args.ckpt, map_location='cpu')
    if 'base_model' in state_dict:
        model_dict = {k.replace('module.', ''): v for k, v in state_dict['base_model'].items()}
    else:
        model_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(model_dict, strict=False)

    base_model = base_model.to(device)
    base_model = nn.DataParallel(base_model)
    base_model.eval()

    # 创建 IPGR 模块
    ipgr = IPGR(base_alpha=args.base_alpha, num_iter=args.num_iter)

    # 评估指标
    chamfer_l1 = ChamferDistanceL1()
    chamfer_l2 = ChamferDistanceL2()

    cdl1_list = []
    cdl2_list = []

    print('\nTesting...')
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            try:
                partial = data[0].to(device)
                gt = data[1].to(device)

                # 模型预测
                pred = base_model(partial)

                # IPGR 后处理
                if not args.no_ipgr:
                    pred = ipgr(pred, partial)

                # 计算 Chamfer Distance
                cdl1 = chamfer_l1(pred, gt).item() * 1000
                cdl2 = chamfer_l2(pred, gt).item() * 10000

                cdl1_list.append(cdl1)
                cdl2_list.append(cdl2)

                if (idx + 1) % 1000 == 0:
                    print(f'[{idx + 1}/{len(test_dataloader)}] '
                          f'CDL1: {np.mean(cdl1_list):.3f}, '
                          f'CDL2: {np.mean(cdl2_list):.3f}')

            except Exception as e:
                print(f'Error at {idx}: {e}')
                continue

    # 结果统计
    cdl1_arr = np.array(cdl1_list)
    cdl2_arr = np.array(cdl2_list)

    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'CDL1:       {cdl1_arr.mean():.4f}')
    print(f'CDL2:       {cdl2_arr.mean():.4f}')
    print(f'Median:     {np.median(cdl2_arr):.4f}')
    print(f'Std:        {cdl2_arr.std():.4f}')
    print(f'< 4.0:      {(cdl2_arr < 4).mean() * 100:.1f}%')
    print('=' * 70)

    if cdl2_arr.mean() < 4.0:
        print('🎉 Target Achieved! (CDL2 < 4.0)')

    return cdl2_arr.mean()


if __name__ == '__main__':
    main()