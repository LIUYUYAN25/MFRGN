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
import time
import shutil
import sys
import torch
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader
import pickle
from torch.utils.tensorboard import SummaryWriter
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
from sample4geo.loss import InfoNCE, InfoNCEMargin, InfoNCEWithEdge, MultiSimilarityLoss
from model.mfrgn_ir import TimmModel, TimmModel_u


def get_parameter_number(model):
    total_num     = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num / 1000000, 'Trainable': trainable_num / 1000000}


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

@dataclass
class Configuration:

    net: str     = 'mfrgn_uavvisloc'
    # [新增，对齐 train_cvact.py] 训练时同步备份模型定义文件
    net_file: str = 'model/mfrgn.py'

    # ── 模型 ──────────────────────────────────────────────────────────────
    sat_model_name: str   = 'convnext_base.fb_in22k_ft_in1k'
    grd_model_name: str   = 'convnext_base.fb_in22k_ft_in1k'
    img_size: int = 256      # 模型输入边长（正方形）
    psm: bool    = True     # 是否使用 PSM 模块

    # ── 数据集 ────────────────────────────────────────────────────────────
    data_folder: str = '../Datasets/UAV_VisLoc_dataset'

    # 场景划分
    train_scene_ids: List[str] = field(
        default_factory=lambda: ['01', '02', '03', '04', '05', '06', '07', '08']
    )
    val_scene_ids: List[str] = field(
        default_factory=lambda: ['09', '10', '11']
    )

    is_polar: bool = False
    image_size_sat = (img_size, img_size)
    img_size_ground = (img_size, img_size)
    # 卫星 patch 裁剪尺寸（在原始卫星图分辨率下）
    sat_patch_size: int = 512

    # ── 训练超参 ──────────────────────────────────────────────────────────
    mixed_precision: bool = True
    seed = 42
    epochs: int  = 15
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0,)

    # ── 评估 ──────────────────────────────────────────────────────────────
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 3
    normalize_features: bool = True

    # ── 优化器 ────────────────────────────────────────────────────────────
    clip_grad = 100.                    # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False    # Gradient Checkpointing

    # ── 学习率 / 调度 ─────────────────────────────────────────────────────
    lr: float          = 0.0001
    scheduler: str     = 'cosine'       # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float      = 0.00001        # only for "polynomial"

    # ── 数据增广 ──────────────────────────────────────────────────────────
    prob_flip: float   = 0.5
    prob_rotate: float = 0.5

    # ── 损失 ──────────────────────────────────────────────────────────────
    label_smoothing: float = 0.1

    # ── 其他 ──────────────────────────────────────────────────────────────
    model_path: str   = 'results_uavvisloc'
    zero_shot: bool   = False
    checkpoint_start  = None
    num_workers: int  = 0 if os.name == 'nt' else 8
    device: str       = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool     = False
    cudnn_deterministic: bool = True


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration()


if __name__ == '__main__':

    # [对齐其他脚本] model_path 变量名与其他三个脚本保持一致（原 save_path）
    model_path = "{}/{}_{}/{}_{}" .format(config.model_path,
                                       config.sat_model_name,
                                       config.grd_model_name,
                                       config.net,
                                       time.strftime("%m-%d-%H-%M-%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))
    # [新增，对齐 train_cvact.py] 同时备份模型定义文件
    shutil.copyfile(config.net_file, "{}/model.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    # [保留] TensorBoard writer（其他脚本无此项，本脚本特有）
    writer = SummaryWriter(log_dir=os.path.join(model_path, 'tensorboard'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#

    print("\nsat_Model: {}".format(config.sat_model_name))
    print("\ngrd_Model: {}".format(config.grd_model_name))

    # [对齐 train_university.py] TimmModel_u 调用方式与 university 保持一致：
    #   university: TimmModel_u(config.model, psm=True, img_size=config.img_size)
    model = TimmModel(config.sat_model_name,
                      config.grd_model_name,
                      config.image_size_sat,
                      config.img_size_ground,
                      psm=config.psm,
                      is_polar=config.is_polar)

    # [新增，对齐 train_university.py] 打印模型内部数据配置（augmentation mean/std 等）
    data_config = model.get_config()
    print(data_config)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    image_size_sat   = config.image_size_sat
    img_size_ground  = config.img_size_ground

    # [新增，对齐其他脚本] Gradient Checkpointing 支持
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    params = get_parameter_number(model)
    print("Total: {} M   Trainable: {} M".format(params['Total'], params['Trainable']))

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))
    print("lr: ", config.lr)
    print("batch size: ", config.batch_size)

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # ------------------ 加载距离字典 ------------------
    dict_path = 'pretrained/MFRGN-pretained/distance_dict/gps_dict_uavvisloc.pkl'
    print(f"Loading distance dict from: {dict_path}")
    with open(dict_path, 'rb') as f:
        raw_dict = pickle.load(f)
    
    # 转换：提取 [(idx, dist), ...] 中的 idx，保证输入给 shuffle 函数的是纯索引
    sim_dict = {k: [neighbor[0] for neighbor in v] for k, v in raw_dict.items()}
    print(f"Loaded distance dict with {len(sim_dict)} anchors.")

    # Transforms
    # 无人机图和卫星 patch 都是俯视正方形，统一使用正方形变换
    sat_transforms_train, ground_transforms_train = get_transforms_train(
        image_size_sat=image_size_sat,
        img_size_ground=img_size_ground,
        mean=mean,
        std=std,
    )

    sat_transforms_val, ground_transforms_val = get_transforms_val(
        image_size_sat=image_size_sat,
        img_size_ground=img_size_ground,
        mean=mean,
        std=std,
    )

    # Train
    train_dataset = UAVVisLocDatasetTrain(
        data_folder=config.data_folder,
        scene_ids=config.train_scene_ids,
        transforms_query=ground_transforms_train,
        transforms_reference=sat_transforms_train,
        prob_flip=config.prob_flip,
        prob_rotate=config.prob_rotate,
        shuffle_batch_size=config.batch_size,
        sat_patch_size=config.sat_patch_size,
    )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)

    # Reference Satellite Images (Val)
    reference_dataset_val = UAVVisLocDatasetEval(
        data_folder=config.data_folder,
        scene_ids=config.val_scene_ids,
        img_type='reference',
        transforms=sat_transforms_val,
        sat_patch_size=config.sat_patch_size,
    )

    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)

    # Query Drone Images (Val)
    query_dataset_val = UAVVisLocDatasetEval(
        data_folder=config.data_folder,
        scene_ids=config.val_scene_ids,
        img_type='query',
        transforms=ground_transforms_val,
        sat_patch_size=config.sat_patch_size,
    )

    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size_eval,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)

    print("Train Images:", len(train_dataset))
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    # [对齐其他脚本] 先构造 loss_fn 再传入 InfoNCE，风格与 cvact/cvusa/university 一致
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    # loss_function = InfoNCEMargin(loss_function=loss_fn, device=config.device)
    loss_function = InfoNCEWithEdge(loss_function=loss_fn, 
                                margin=0.1, 
                                edge_weight=0.1, 
                                device=config.device)
    loss_function = MultiSimilarityLoss(alpha=2.0, 
                                        beta=50.0, 
                                        base=0.5, 
                                        margin=0.1)
    # 将模型中的 logit_scale 的 required_grad 设为 False (因为 MS Loss 自带超参，不再需要学习温度系数)
    model.logit_scale.requires_grad = False
    if hasattr(model, 'module'):
        model.module.logit_scale.requires_grad = False

    # [修改] torch.cuda.amp.GradScaler 已在 PyTorch >= 2.0 中弃用
    # 改用新 API: torch.amp.GradScaler(device=...) 
    # 其他三个脚本迁移方式（供参考）:
    #   旧: scaler = GradScaler(init_scale=2.**10)
    #   新: scaler = torch.amp.GradScaler(device='cuda', init_scale=2.**10)
    if config.mixed_precision:
        scaler = torch.amp.GradScaler(device=config.device, init_scale=2.**10)
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # Optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    # [新增，对齐其他脚本] 增加 decay_exclue_bias 分组权重衰减，与 cvact/cvusa/university 结构对齐
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    # [对齐其他脚本] 变量名从 total_steps 改为 train_steps，与其他三个脚本一致
    train_steps  = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    # [对齐其他脚本] 打印 Warmup/Train 步数信息
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_val,
                           query_dataloader=query_dataloader_val,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           is_dual=True,
                           cleanup=True)

    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    best_score = 0

    for epoch in range(1, config.epochs + 1):

        # ============= 每个 Epoch 动态构造包含难负样本的 Batch =============
        # neighbour_select=4: 因为你平均只有 5.6 个邻居，选 4 个放进 Batch 最合适
        # neighbour_range=16: 在最近的 16 个样本里去选
        train_dataset.shuffle(sim_dict=sim_dict, 
                              neighbour_select=4, 
                              neighbour_range=16)
        # ===============================================================

        print("\n{}[Epoch: {}/{}]{}".format(30*"-", epoch, config.epochs, 30*"-"))
        s2 = time.time()

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        # [对齐其他脚本] 打印格式统一使用 .format()
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6e}".format(epoch,
                                                                    train_loss,
                                                                    optimizer.param_groups[0]['lr']))

        s3 = time.time()
        print('train: ', (s3 - s2) / 60, 'min')

        # Evaluate
        # [对齐 train_university.py] 条件与 university 一致：每 n 轮或最后一轮都评估
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_val,
                               query_dataloader=query_dataloader_val,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               is_dual=True,
                               is_autocast=True,
                               cleanup=True)

            # [保留] TensorBoard 记录 Recall@1（其他脚本无此项，本脚本特有）
            writer.add_scalar('Eval/Recall@1', r1_test, epoch)

            if r1_test > best_score:
                best_score = r1_test

                # [对齐其他脚本] 显式判断 DataParallel 再保存，与其他三个脚本完全一致
                # （原代码用 hasattr(model, 'module') 判断，改为与其他脚本相同的方式）
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

    # Save final weights
    # [对齐其他脚本] 最终权重保存逻辑与其他三个脚本完全对齐
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    print("\nTraining finished. Best Recall@1: {:.4f}".format(best_score))
    print("Weights saved to: {}".format(model_path))
    writer.close()