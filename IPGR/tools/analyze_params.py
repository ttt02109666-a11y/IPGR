"""
IPGR 参数敏感性分析

测试不同 alpha 和 iteration 组合的效果

使用方法:
    python analyze_params.py --config cfgs/xxx.yaml --ckpt path/to/ckpt.pth
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.getcwd())

from tools import builder
from utils.config import cfg_from_yaml_file
from easydict import EasyDict
from extensions.chamfer_dist import ChamferDistanceL2


class IPGR(nn.Module):
    """Iterative Partial-Guided Refinement"""

    def __init__(self, base_alpha=0.05, num_iter=2):
        super().__init__()
        self.base_alpha = base_alpha
        self.num_iter = num_iter

    def forward(self, pred, partial):
        refined = pred.clone()
        B, N, _ = pred.shape
        batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand(-1, N)

        for i in range(self.num_iter):
            dist = torch.cdist(refined, partial)
            min_dist, min_idx = dist.min(dim=-1)
            nearest = partial[batch_idx, min_idx]

            dist_norm = min_dist / (min_dist.max(dim=-1, keepdim=True)[0] + 1e-6)
            alpha = self.base_alpha * (2.0 - dist_norm)

            refined = refined + alpha.unsqueeze(-1) * (nearest - refined)

        return refined


def test_with_params(base_model, test_dataloader, alpha, num_iter, device, num_samples=500):
    """用指定参数测试"""

    ipgr = IPGR(base_alpha=alpha, num_iter=num_iter)
    chamfer_l2 = ChamferDistanceL2()

    cdl2_list = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if idx >= num_samples:
                break

            partial = data[0].to(device)
            gt = data[1].to(device)

            # 模型预测
            output = base_model(partial)
            if isinstance(output, tuple):
                pred = output[-1]
            else:
                pred = output

            # IPGR 后处理
            pred_refined = ipgr(pred, partial)

            # 计算 CDL2
            cdl2 = chamfer_l2(pred_refined, gt).item() * 10000
            cdl2_list.append(cdl2)

    return np.mean(cdl2_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=500)
    args = parser.parse_args()

    print('=' * 60)
    print('IPGR Parameter Sensitivity Analysis')
    print('=' * 60)

    # 加载配置
    config = cfg_from_yaml_file(args.config)
    config = EasyDict(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    class Args:
        distributed = False
        num_workers = 4

    _, test_dataloader = builder.dataset_builder(Args(), config.dataset.test)

    # 构建模型
    base_model = builder.model_builder(config.model)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    if 'base_model' in state_dict:
        model_dict = {k.replace('module.', ''): v for k, v in state_dict['base_model'].items()}
    else:
        model_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(model_dict, strict=False)
    base_model = base_model.to(device)
    base_model = nn.DataParallel(base_model)
    base_model.eval()

    # 测试参数范围
    alphas = [0.01, 0.03, 0.05, 0.07, 0.10]
    iters = [1, 2, 3, 4, 5]

    # 存储结果
    results = {}

    print(f'\nTesting {len(alphas) * len(iters)} parameter combinations...\n')

    for alpha in alphas:
        for num_iter in iters:
            print(f'Testing alpha={alpha}, iter={num_iter}...', end=' ')
            cdl2 = test_with_params(base_model, test_dataloader, alpha, num_iter, device, args.num_samples)
            results[(alpha, num_iter)] = cdl2
            print(f'CDL2={cdl2:.3f}')

    # 打印结果表格
    print('\n' + '=' * 60)
    print('RESULTS: Parameter Sensitivity')
    print('=' * 60)

    # 表头
    print(f'{"alpha":<8}', end='')
    for num_iter in iters:
        print(f'iter={num_iter:<6}', end='')
    print()
    print('-' * 45)

    # 数据行
    best_cdl2 = float('inf')
    best_params = None

    for alpha in alphas:
        print(f'{alpha:<8}', end='')
        for num_iter in iters:
            cdl2 = results[(alpha, num_iter)]
            print(f'{cdl2:<10.3f}', end='')
            if cdl2 < best_cdl2:
                best_cdl2 = cdl2
                best_params = (alpha, num_iter)
        print()

    print('-' * 45)
    print(f'\n✅ Best: alpha={best_params[0]}, iter={best_params[1]}, CDL2={best_cdl2:.3f}')

    # 保存结果到文件
    with open('param_sensitivity_results.txt', 'w') as f:
        f.write('IPGR Parameter Sensitivity Analysis\n')
        f.write('=' * 45 + '\n')
        f.write(f'{"alpha":<8}')
        for num_iter in iters:
            f.write(f'iter={num_iter:<6}')
        f.write('\n')
        for alpha in alphas:
            f.write(f'{alpha:<8}')
            for num_iter in iters:
                f.write(f'{results[(alpha, num_iter)]:<10.3f}')
            f.write('\n')
        f.write(f'\nBest: alpha={best_params[0]}, iter={best_params[1]}, CDL2={best_cdl2:.3f}\n')

    print('\n结果已保存到 param_sensitivity_results.txt')

    return results


if __name__ == '__main__':
    main()