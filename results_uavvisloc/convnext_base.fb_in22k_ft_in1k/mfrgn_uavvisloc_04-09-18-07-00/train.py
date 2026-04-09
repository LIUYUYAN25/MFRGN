"""
train_uavvisloc.py
==================
使用 UAV-VisLoc 数据集训练 MFRGN 模型。

CSV 格式（自动识别，有表头）:
  num | filename | date | lat | lon | height | Omega | Kappa | Phi1 | Phi2

坐标转换: lat/lon(WGS84) → 卫星图像素坐标（通过 rasterio 读取 GeoTIFF 仿射变换）

模型: TimmModel_u（共享权重），适合无人机 vs 卫星同视角匹配。

场景划分（默认）:
  训练: 01~08    测试: 09~11
"""

import os
import sys
import time
import shutil
import torch
from dataclasses import dataclass, field
from typing import List
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from sample4geo.dataset.uavvisloc import UAVVisLocDatasetTrain, UAVVisLocDatasetEval
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.cvusa_and_cvact import evaluate
from sample4geo.loss import InfoNCE
from model.mfrgn import TimmModel_u


def get_parameter_number(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total / 1e6, 'Trainable': trainable / 1e6}


# ══════════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════════

@dataclass
class Configuration:

    net: str = 'mfrgn_uavvisloc'

    # ── 模型 ───────────────────────────────────────────────────────
    model: str  = 'convnext_base.fb_in22k_ft_in1k'
    img_size: int = 256      # 模型输入边长（正方形）
    psm: bool = True

    # ── 数据集 ─────────────────────────────────────────────────────
    data_folder: str = '../Datasets/UAV_VisLoc_dataset'

    # 场景划分
    train_scene_ids: List[str] = field(
        default_factory=lambda: ['01','02','03','04','05','06','07','08']
    )
    val_scene_ids: List[str] = field(
        default_factory=lambda: ['09','10','11']
    )

    # 卫星 patch 裁剪尺寸（在原始卫星图分辨率下）
    # 建议：飞行高度 ~400m 时，卫星图地面分辨率约 0.5~1m/px，
    # 无人机拍摄覆盖约 200m×200m，对应 200~400px，
    # 取 512 保留足够上下文。
    sat_patch_size: int = 512

    # ── 训练超参 ────────────────────────────────────────────────────
    mixed_precision: bool = True
    seed: int = 42
    epochs: int = 30
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0, 1, 2, 3)

    # ── 评估 ────────────────────────────────────────────────────────
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 3
    normalize_features: bool = True

    # ── 优化器 ──────────────────────────────────────────────────────
    lr: float = 0.0001
    scheduler: str = 'cosine'
    warmup_epochs: int = 1
    lr_end: float = 0.00001
    clip_grad: float = 100.

    # ── 数据增广 ────────────────────────────────────────────────────
    prob_flip: float   = 0.5
    prob_rotate: float = 0.5   # 卫星 patch 旋转（俯视图方向不重要，旋转增广合理）

    # ── 损失 ────────────────────────────────────────────────────────
    label_smoothing: float = 0.1

    # ── 其他 ────────────────────────────────────────────────────────
    model_path: str = 'results_uavvisloc'
    checkpoint_start: str = None
    zero_shot: bool = False
    num_workers: int = 0 if os.name == 'nt' else 8
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True


# ══════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════

config = Configuration()

if __name__ == '__main__':

    save_path = os.path.join(
        config.model_path, config.model,
        f"{config.net}_{time.strftime('%m-%d-%H-%M-%S')}"
    )
    os.makedirs(save_path, exist_ok=True)
    shutil.copyfile(os.path.basename(__file__), os.path.join(save_path, 'train.py'))
    sys.stdout = Logger(os.path.join(save_path, 'log.txt'))

    setup_system(config.seed, config.cudnn_benchmark, config.cudnn_deterministic)

    # ── 模型 ─────────────────────────────────────────────────────────
    print(f"\nModel: {config.model}")
    model = TimmModel_u(
        model_name=config.model,
        img_size=config.img_size,
        psm=config.psm,
    )

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img_size_tuple = (config.img_size, config.img_size)

    if config.checkpoint_start:
        print(f"从 checkpoint 恢复: {config.checkpoint_start}")
        model.load_state_dict(
            torch.load(config.checkpoint_start), strict=False
        )

    p = get_parameter_number(model)
    print(f"Total: {p['Total']:.2f}M   Trainable: {p['Trainable']:.2f}M")

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(config.device)

    print(f"img_size: {img_size_tuple}  sat_patch_size: {config.sat_patch_size}")
    print(f"LR: {config.lr}  batch: {config.batch_size}\n")

    # ── 数据变换 ──────────────────────────────────────────────────────
    # 无人机图和卫星 patch 都是俯视正方形，统一用正方形变换
    sat_tr_train, drone_tr_train = get_transforms_train(
        image_size_sat=img_size_tuple,
        img_size_ground=img_size_tuple,
        mean=mean, std=std,
    )
    sat_tr_val, drone_tr_val = get_transforms_val(
        image_size_sat=img_size_tuple,
        img_size_ground=img_size_tuple,
        mean=mean, std=std,
    )

    # ── 数据集 ───────────────────────────────────────────────────────
    train_dataset = UAVVisLocDatasetTrain(
        data_folder=config.data_folder,
        scene_ids=config.train_scene_ids,
        transforms_query=drone_tr_train,
        transforms_reference=sat_tr_train,
        prob_flip=config.prob_flip,
        prob_rotate=config.prob_rotate,
        shuffle_batch_size=config.batch_size,
        sat_patch_size=config.sat_patch_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, shuffle=True,
        pin_memory=True, drop_last=True,
    )

    ref_dataset_val = UAVVisLocDatasetEval(
        data_folder=config.data_folder, scene_ids=config.val_scene_ids,
        img_type='reference', transforms=sat_tr_val,
        sat_patch_size=config.sat_patch_size,
    )
    ref_loader_val = DataLoader(
        ref_dataset_val, batch_size=config.batch_size_eval,
        num_workers=config.num_workers, shuffle=False, pin_memory=True,
    )

    qry_dataset_val = UAVVisLocDatasetEval(
        data_folder=config.data_folder, scene_ids=config.val_scene_ids,
        img_type='query', transforms=drone_tr_val,
        sat_patch_size=config.sat_patch_size,
    )
    qry_loader_val = DataLoader(
        qry_dataset_val, batch_size=config.batch_size_eval,
        num_workers=config.num_workers, shuffle=False, pin_memory=True,
    )

    print(f"训练样本: {len(train_dataset)}")
    print(f"验证 Reference: {len(ref_dataset_val)}  Query: {len(qry_dataset_val)}")

    # ── 损失 / 优化器 / 调度 ─────────────────────────────────────────
    loss_function = InfoNCE(
        torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing),
        device=config.device
    )
    scaler    = GradScaler(init_scale=2.**10) if config.mixed_precision else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    total_steps  = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    if config.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, total_steps, warmup_steps)
    elif config.scheduler == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, total_steps, config.lr_end, 1.5, warmup_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    print(f"Scheduler: {config.scheduler}  "
          f"Warmup: {warmup_steps}步  Total: {total_steps}步")

    # ── Zero-shot 评估 ────────────────────────────────────────────────
    if config.zero_shot:
        print(f"\n{'─'*30}[Zero Shot]{'─'*30}")
        evaluate(config=config, model=model,
                 reference_dataloader=ref_loader_val,
                 query_dataloader=qry_loader_val,
                 ranks=[1,5,10], step_size=1000,
                 is_dual=True, is_autocast=True, cleanup=True)

    # ── 训练 ─────────────────────────────────────────────────────────
    best_score = 0.0

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'─'*30}[Epoch {epoch}/{config.epochs}]{'─'*30}")
        t0 = time.time()

        train_loss = train(
            train_config=config, model=model,
            dataloader=train_loader, loss_function=loss_function,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        print(f"Loss={train_loss:.4f}  "
              f"LR={optimizer.param_groups[0]['lr']:.2e}  "
              f"Time={(time.time()-t0)/60:.1f}min")

        if epoch % config.eval_every_n_epoch == 0 or epoch == config.epochs:
            print(f"\n{'─'*30}[Evaluate]{'─'*30}")
            r1 = evaluate(
                config=config, model=model,
                reference_dataloader=ref_loader_val,
                query_dataloader=qry_loader_val,
                ranks=[1,5,10], step_size=1000,
                is_dual=True, is_autocast=True, cleanup=True,
            )
            state = (model.module.state_dict()
                     if hasattr(model, 'module') else model.state_dict())
            if r1 > best_score:
                best_score = r1
                torch.save(state,
                           os.path.join(save_path, f'weights_e{epoch}_{r1:.4f}.pth'))
                print(f"  ★ 新最佳 Recall@1: {best_score:.4f}")

    state = (model.module.state_dict()
             if hasattr(model, 'module') else model.state_dict())
    torch.save(state, os.path.join(save_path, 'weights_end.pth'))
    print(f"\n训练完成。最佳 Recall@1: {best_score:.4f}")
    print(f"权重保存: {save_path}")
