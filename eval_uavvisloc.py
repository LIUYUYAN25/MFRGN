"""
eval_uavvisloc.py
=================
对 UAV-VisLoc 数据集进行评估，输出 Recall@1/5/10。

CSV 格式（自动识别，有表头）:
  num | filename | date | lat | lon | height | Omega | Kappa | Phi1 | Phi2

使用:
  python eval_uavvisloc.py
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from sample4geo.dataset.uavvisloc import UAVVisLocDatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvusa_and_cvact import evaluate
from model.mfrgn import TimmModel_u


@dataclass
class Configuration:

    model: str  = 'convnext_base.fb_in22k_ft_in1k'
    img_size: int = 256
    psm: bool = True

    data_folder: str = '../Datasets/UAV_VisLoc_dataset'

    # 评估场景（默认使用测试场景 09/10/11）
    val_scene_ids: List[str] = field(
        default_factory=lambda: ['09', '10', '11']
    )

    sat_patch_size: int = 512

    # 评估超参
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0, 1)
    normalize_features: bool = True

    checkpoint_start: str = 'results_uavvisloc/weights_end.pth'

    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Configuration()

if __name__ == '__main__':

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
        print(f"加载权重: {config.checkpoint_start}")
        model.load_state_dict(
            torch.load(config.checkpoint_start, map_location='cpu'), strict=False
        )

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(config.device)

    print(f"评估场景: {config.val_scene_ids}")
    print(f"sat_patch_size: {config.sat_patch_size}  img_size: {img_size_tuple}\n")

    sat_tr_val, drone_tr_val = get_transforms_val(
        image_size_sat=img_size_tuple,
        img_size_ground=img_size_tuple,
        mean=mean, std=std,
    )

    ref_dataset = UAVVisLocDatasetEval(
        data_folder=config.data_folder,
        scene_ids=config.val_scene_ids,
        img_type='reference',
        transforms=sat_tr_val,
        sat_patch_size=config.sat_patch_size,
    )
    ref_loader = DataLoader(
        ref_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, shuffle=False, pin_memory=True,
    )

    qry_dataset = UAVVisLocDatasetEval(
        data_folder=config.data_folder,
        scene_ids=config.val_scene_ids,
        img_type='query',
        transforms=drone_tr_val,
        sat_patch_size=config.sat_patch_size,
    )
    qry_loader = DataLoader(
        qry_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, shuffle=False, pin_memory=True,
    )

    print(f"Reference: {len(ref_dataset)}  Query: {len(qry_dataset)}")

    print(f"\n{'─'*30}[UAV-VisLoc Eval]{'─'*30}")
    r1 = evaluate(
        config=config, model=model,
        reference_dataloader=ref_loader,
        query_dataloader=qry_loader,
        ranks=[1, 5, 10], step_size=1000,
        is_dual=True, is_autocast=True, cleanup=True,
    )
    print(f"\n最终 Recall@1: {r1:.4f}")


# ══════════════════════════════════════════════════════════════════
#  分场景评估（可选调用）
# ══════════════════════════════════════════════════════════════════

def eval_per_scene(config: Configuration = None):
    """
    对每个场景单独输出 Recall@1，方便定位难易场景。

    调用方式（在脚本末尾加）:
        if __name__ == '__main__':
            ...
            eval_per_scene(config)
    """
    if config is None:
        config = Configuration()

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img_size_tuple = (config.img_size, config.img_size)

    model = TimmModel_u(config.model, img_size=config.img_size, psm=config.psm)
    if config.checkpoint_start:
        model.load_state_dict(
            torch.load(config.checkpoint_start, map_location='cpu'), strict=False
        )
    model = model.to(config.device)

    sat_tr_val, drone_tr_val = get_transforms_val(
        img_size_tuple, img_size_tuple, mean, std
    )

    results = {}
    for scene_id in config.val_scene_ids:
        ref_ds = UAVVisLocDatasetEval(
            config.data_folder, [scene_id], 'reference',
            sat_tr_val, config.sat_patch_size
        )
        qry_ds = UAVVisLocDatasetEval(
            config.data_folder, [scene_id], 'query',
            drone_tr_val, config.sat_patch_size
        )
        if len(ref_ds) == 0:
            continue

        ref_ld = DataLoader(ref_ds, config.batch_size, num_workers=config.num_workers,
                            shuffle=False, pin_memory=True)
        qry_ld = DataLoader(qry_ds, config.batch_size, num_workers=config.num_workers,
                            shuffle=False, pin_memory=True)

        print(f"\n{'─'*25}[Scene {scene_id}]{'─'*25}")
        r1 = evaluate(
            config=config, model=model,
            reference_dataloader=ref_ld, query_dataloader=qry_ld,
            ranks=[1, 5, 10], step_size=500,
            is_dual=True, is_autocast=True, cleanup=True,
        )
        results[scene_id] = r1

    print("\n\n各场景 Recall@1:")
    for sid, r in results.items():
        print(f"  场景 {sid}: {r:.4f}")
    if results:
        print(f"  平均: {sum(results.values())/len(results):.4f}")
    return results
