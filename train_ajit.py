"""
train_ajit.py — A-Point-Set-Generation 학습 로직

이 파일은 모델 학습(train)과 검증(val)의 핵심 로직을 담당한다.
baseline_main.py가 이 파일의 train() 함수를 호출하여 학습을 시작한다.

주요 흐름:
  1. train(): 학습 루프 — 매 epoch마다 학습 데이터로 가중치를 갱신하고,
     검증 데이터로 성능을 측정한 뒤 체크포인트를 저장한다.
  2. val(): 검증 루프 — 학습된 모델로 검증 데이터의 loss와 mIoU를 계산한다.
  3. write_json_to_file() / read_json_file(): 학습 이력을 JSON 파일로 저장/불러온다.
     이력은 학습 재개(use_checkpoint) 시 마지막으로 학습한 epoch를 파악하는 데 사용된다.

왜 이 파일이 분리되어 있는가:
  학습 로직과 엔트리포인트(경로 설정, 데이터 로더 생성 등)를 분리하면
  코드를 더 쉽게 유지보수할 수 있고, 다른 학습 스크립트에서도 재사용 가능하다.
"""

from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# PartDataset는 원본 ShapeNet 데이터용 클래스이며, 현재 NIA29 데이터에는 사용하지 않는다.
# 다만 train_ajit.py가 import 시도를 하므로 호환성을 위해 유지 (cv2 필요).
from datasets import PartDataset
import torch.nn.functional as F
import torch.cuda as cuda
from pic2points_model import pic2points
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch
# ChamferDistance: 예측 포인트 클라우드와 정답 포인트 클라우드 간의 거리를 측정하는 손실 함수.
# Linux에서는 pytorch3d가 제공하는 CUDA 가속 버전을, Windows에서는 저장소 내 chamfer_distance.py(순수 PyTorch)를 사용한다.
from chamfer_distance import ChamferDistance
from data_loader import XDataset, get_loader
from split_data import read_from_file, write_to_file, split_data
import json, os
import time
import datetime

# calculate_3d_miou: 예측 포인트 클라우드와 정답 포인트 클라우드 간의 3D IoU를 계산한다.
# IoU(Intersection over Union)는 두 3D 박스가 얼마나 겹치는지를 0~1 사이 값으로 나타낸다.
from metrics import calculate_3d_miou
#from emd import EMDLoss  # Earth Mover's Distance — Chamfer Distance의 대안 손실 함수 (현재 미사용)


def train(model: nn.Module, train_loader, val_loader, chamferDist, num_epochs, lr, model_name="Baseline", use_checkpoint=False, device=None):
    """
    모델 학습 루프.

    매 epoch마다:
      1. 학습 데이터로 forward → loss 계산 → backward → 가중치 갱신
      2. 검증 데이터로 현재 성능(loss, mIoU) 측정
      3. 성능이 개선되면 best 체크포인트 저장
      4. 항상 latest 체크포인트 저장 (학습 재개용)
      5. 학습 이력을 JSON 파일에 기록

    Args:
        model: 학습할 포인트 클라우드 생성 모델 (pic2points)
        train_loader: 학습 데이터 DataLoader
        val_loader: 검증 데이터 DataLoader
        chamferDist: Chamfer Distance 손실 함수 인스턴스
        num_epochs: 총 학습 epoch 수
        lr: 학습률 (Adam optimizer)
        model_name: 체크포인트 파일명에 사용할 모델 이름
        use_checkpoint: 기존 체크포인트에서 이어서 학습할지 여부

    Returns:
        training_losses: epoch별 학습 loss 리스트
        validation_losses: epoch별 검증 loss 리스트
        best_model: 검증 loss가 가장 낮았을 때의 모델
    """

    # 왜 device를 분기하는가: CUDA가 없는 환경(Windows CPU 디버깅 등)에서도
    # .cuda() 호출로 인한 오류가 발생하지 않도록 한다.
    # device 파라미터가 전달된 경우(예: --cpu 플래그) 해당 디바이스를 우선 사용한다.
    if device is not None:
        gpu_or_cpu = device
    elif torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    # Adam optimizer를 사용하는 이유:
    # SGD에 비해 학습률을 각 파라미터마다 자동으로 조절(adaptive)하여
    # 포인트 클라우드 회귀 같은 작업에서 더 안정적으로 수렴한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 학습 이력(JSON 파일)을 읽어와서 이전에 학습한 epoch 수를 확인한다.
    # 이력 파일이 없으면 빈 리스트를 반환받아 처음부터 학습을 시작한다.
    experiment_data = read_json_file(model_name)

    # use_checkpoint=True이면 이전에 학습한 epoch 수만큼 건너뛰고 이어서 학습한다.
    # use_checkpoint=False이면 처음부터 다시 학습한다.
    if use_checkpoint:
        num_of_epochs_already_performed = len(experiment_data)
    else:
        num_of_epochs_already_performed = 0

    # 이전 학습 이력이 있는 경우, 이력과 체크포인트를 불러와서 이어서 학습한다.
    if num_of_epochs_already_performed != 0:
        # 체크포인트 파일(.pt)에서 모델 전체를 로드한다.
        # 왜 torch.load(model)를 쓰는가: 이 프로젝트는 모델 전체 객체를 저장하는 방식을 사용한다.
        # (state_dict만 저장하는 방식보다 간단하지만, 코드 구조가 바뀌면 호환성 문제가 생길 수 있다.)
        model = torch.load(model_name + ".pt", weights_only=False)
        training_losses = [e['training_loss'] for e in experiment_data]
        training_mious = [e['training_miou'] for e in experiment_data]
        validation_losses = [e['val_loss'] for e in experiment_data]
        validation_mious = [e['val_miou'] for e in experiment_data]
        # best_val_performance: 지금까지 검증 loss 중 최소값.
        # 이 값보다 더 낮은 loss가 나와야 best 체크포인트를 갱신한다.
        best_val_performance = min(validation_losses)
    else:
        # 처음 학습하는 경우, 모든 이력 리스트를 빈 상태로 초기화한다.
        training_losses = []
        training_mious = []
        validation_losses = []
        validation_mious = []
        # float('inf'): 무한대. 어떤 유한한 loss값보다도 크므로 첫 epoch에서 무조건 best가 갱신된다.
        best_val_performance = float('inf')

    print(f"Continuing training from Epoch {num_of_epochs_already_performed}")

    # 모델을 학습 모드로 전환한다.
    # 왜 train()을 호출하는가: Dropout, BatchNorm 등은 학습/추론 시 다르게 동작한다.
    # train()을 호출하면 학습 모드로 설정되어, 정확한 gradient 계산이 가능해진다.
    model.train()

    # 이미 학습한 epoch 이후부터 num_epochs까지 학습을 진행한다.
    for epoch in range(num_of_epochs_already_performed, num_epochs):
        epoch_start_time = time.time()
        training_loss = 0.
        training_miou = 0.
        batch_count = len(train_loader)

        # ── 학습 루프: 매 배치마다 forward → loss → backward → step ──
        for i, (image, point_cloud) in enumerate(train_loader):
            batch_start_time = time.time()

            # Variable로 감싸는 이유: 구 버전 PyTorch(0.4 이전) 호환성.
            # 최신 PyTorch에서는 tensor 자체가 autograd를 지원하므로 Variable은 사실상 불필요하지만,
            # 원본 코드와의 호환성을 위해 유지한다.
            image, point_cloud = Variable(image), Variable(point_cloud)

            # 데이터를 GPU(또는 CPU)로 이동시킨다.
            # 왜 float()을 호출하는가: DataLoader가 반환하는 텐서가 double 타입일 수 있어,
            # 모델의 가중치(float32)와 타입을 맞추기 위해 float32로 변환한다.
            image = image.float().to(device=gpu_or_cpu)
            point_cloud = point_cloud.float().to(device=gpu_or_cpu)

            # forward pass: 이미지를 입력으로 3D 포인트 클라우드를 예측한다.
            pred = model(image)

            # Chamfer Distance 손실 계산:
            # dist[0]: 정답→예측 방향 거리 (각 정답 점에서 가장 가까운 예측 점까지의 거리)
            # dist[1]: 예측→정답 방향 거리 (각 예측 점에서 가장 가까운 정답 점까지의 거리)
            # 두 방향의 평균 거리를 더해서 양방향 거리를 만든다.
            # 왜 100.0으로 나누는가: Chamfer Distance 값이 클 수 있어 학습률과의 균형을 위해 정규화.
            dist = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist[0]) + torch.mean(dist[1])) / 100.0  # 정규화

            # mIoU(Mean IoU): 예측과 정답 포인트 클라우드의 3D 바운딩 박스가 얼마나 겹치는지를 측정.
            # loss와 함께 모니터링하여 학습 진행 상황을 파악한다.
            miou = calculate_3d_miou(pred, point_cloud)

            # 역전파 전에 이전 배치의 gradient를 초기화한다.
            # 왜 zero_grad()가 필요한가: PyTorch는 gradient를 누적(accumulation)하기 때문에,
            # 초기화하지 않으면 이전 배치의 gradient가 더해져서 잘못된 가중치 갱신이 일어난다.
            optimizer.zero_grad()

            # 역전파: loss에 대한 각 파라미터의 gradient를 계산한다.
            loss.backward()

            # 가중치 갱신: 계산된 gradient로 모델의 가중치를 업데이트한다.
            optimizer.step()

            # .item()으로 tensor에서 Python float 값을 추출한다.
            # tensor 자체를 더하면 computational graph가 누적되어 메모리 누수가 발생한다.
            training_loss += loss.item()
            training_miou += miou.item()

            # 진행 상황 출력: 현재 epoch, 배치, loss, mIoU, 경과 시간을 표시한다.
            elapsed_time = time.time() - epoch_start_time
            time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            print(f"Epoch: [{epoch+1}/{num_epochs}] | "
                  f"Batch: [{i+1}/{batch_count}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"mIoU: {miou.item():.4f} | "
                  f"Time: {time_str}")

        # epoch 전체의 평균 loss와 mIoU를 계산한다.
        avg_training_loss = training_loss / len(train_loader)
        avg_training_miou = training_miou / len(train_loader)

        # 검증 데이터로 현재 모델의 성능을 평가한다.
        val_loss, val_miou = val(model, chamferDist, val_loader, device=gpu_or_cpu)

        # 이력 리스트에 현재 epoch의 결과를 추가한다.
        training_losses.append(avg_training_loss)
        training_mious.append(avg_training_miou)
        validation_losses.append(val_loss)
        validation_mious.append(val_miou)

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training - Loss: {avg_training_loss:.4f}, mIoU: {avg_training_miou:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")
        print(f"Time: {str(datetime.timedelta(seconds=int(epoch_time)))}\n")

        # best 체크포인트 저장: 검증 loss가 이전 최소값보다 낮으면 best 모델을 갱신한다.
        # 왜 loss를 기준으로 하는가: mIoU는 이산적이고 변동이 큰 반면,
        # loss(Chamfer Distance)는 연속적이어서 더 안정적인 기준이 된다.
        if val_loss < best_val_performance:
            best_val_performance = val_loss
            # 모델 전체 객체를 저장한다. (state_dict가 아닌 전체 모델)
            torch.save(model, f"best-{model_name}.pt")
            print(f"New best model saved with validation loss: {val_loss:.4f}")

        # latest 체크포인트 저장: 매 epoch마다 항상 저장한다.
        # 학습이 중단되더라도 이 파일에서 이어서 학습할 수 있다.
        torch.save(model, f"{model_name}.pt")

        # epoch 결과를 JSON 이력 파일에 기록한다.
        # 왜 JSON을 쓰는가: 텍스트 형식이라 사람이 읽을 수 있고,
        # 학습 곡선을 나중에 시각화할 수 있다.
        epoch_data = {
            'training_loss': avg_training_loss,
            'training_miou': avg_training_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'epoch': epoch + 1,
            'time': str(datetime.timedelta(seconds=int(epoch_time)))
        }
        experiment_data.append(epoch_data)
        write_json_to_file(model_name, experiment_data)

    # 학습이 끝나면 best 모델을 로드하여 반환한다.
    # 왜 map_location을 안 쓰는가: 학습 환경과 동일한 환경에서 로드하므로 불필요.
    # (eval.py, testing.py 등에서는 CPU/GPU 환경이 다를 수 있어 map_location을 사용한다.)
    return training_losses, validation_losses, torch.load(f"best-{model_name}.pt", weights_only=False)


def val(model, chamferDist, val_loader, device=None):
    """
    검증 데이터로 모델 성능을 평가한다.

    학습 데이터로 가중치를 갱신하지 않고, 검증 데이터로 일반화 성능을 측정한다.
    매 epoch마다 호출되어 과적합(overfitting) 여부를 모니터링한다.

    Args:
        model: 평가할 모델
        chamferDist: Chamfer Distance 손실 함수 인스턴스
        val_loader: 검증 데이터 DataLoader
        device: 강제로 사용할 디바이스 (None이면 자동 선택)

    Returns:
        avg_val_loss: 검증 데이터의 평균 loss
        avg_val_miou: 검증 데이터의 평균 mIoU
    """
    # 모델을 추론 모드로 전환한다.
    # 왜 eval()을 호출하는가: Dropout을 끄고, BatchNorm을 추론 통계값을 사용하게 한다.
    # 학습 시의 무작위성을 제거하여 동일한 입력에 대해 항상 동일한 출력을 얻는다.
    model.eval()
    total_val_loss = 0
    total_val_miou = 0

    # device가 전달된 경우 해당 디바이스를 우선 사용한다 (--cpu 플래그 등).
    if device is not None:
        gpu_or_cpu = device
    elif torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    # torch.no_grad(): gradient 계산을 비활성화한다.
    # 왜 사용하는가: 검증 시에는 가중치를 갱신하지 않으므로 gradient가 필요 없다.
    # gradient 계산을 생략하면 메모리 사용량과 계산 시간이 크게 줄어든다.
    with torch.no_grad():
        for i, (image, point_cloud) in enumerate(val_loader):
            image, point_cloud = Variable(image), Variable(point_cloud)
            image = image.float().to(device=gpu_or_cpu)
            point_cloud = point_cloud.float().to(device=gpu_or_cpu)

            pred = model(image)

            # 학습 때와 동일한 방식으로 loss와 mIoU를 계산한다.
            dist = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist[0]) + torch.mean(dist[1])) / 100.0

            miou = calculate_3d_miou(pred, point_cloud)

            total_val_loss += loss.item()
            total_val_miou += miou.item()

    # 검증이 끝나면 다시 학습 모드로 전환한다.
    # 왜 다시 train()을 호출하는가: val()이 끝난 후에도 학습 루프가 계속되므로,
    # Dropout과 BatchNorm이 학습 모드로 돌아가야 한다.
    model.train()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_miou = total_val_miou / len(val_loader)

    return avg_val_loss, avg_val_miou


def write_json_to_file(path, data):
    """학습 이력 데이터를 JSON 파일로 저장한다."""
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def read_json_file(path):
    """학습 이력 JSON 파일을 읽어온다. 파일이 없으면 빈 리스트를 반환한다."""
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# 아래는 개선된 버전의 train/val 코드이지만 현재 활성화되지 않는다.
# 주요 차이점:
#   - PT&logs/ 디렉토리에 학습 번호별로 체크포인트와 로그를 체계적으로 저장
#   - 5 epoch마다 주기적 체크포인트 저장 (학습 중단 시 복구 가능)
#   - state_dict 기반 저장 (모델 전체 저장 대신, 더 가볍고 호환성이 좋음)
# 활성화하려면 위쪽 활성 코드를 주석 처리하고 아래 코드의 주석을 해제하면 된다.
# ──────────────────────────────────────────────────────────────────────────────

"""
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda as cuda
from pic2points_model import pic2points
from torch.nn.parallel import DataParallel
import json
import time
import datetime
from metrics import calculate_3d_miou

def get_save_dir():
    base_path = "PT&logs"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return os.path.join(base_path, "train1")

    existing_dirs = [d for d in os.listdir(base_path) if d.startswith("train")]
    if not existing_dirs:
        return os.path.join(base_path, "train1")

    max_num = max([int(d.replace("train", "")) for d in existing_dirs])
    next_dir = os.path.join(base_path, f"train{max_num + 1}")
    return next_dir

def train(model: nn.Module, train_loader, val_loader, chamferDist, num_epochs, lr, model_name="Baseline", use_checkpoint=False):

    # Create save directory
    save_dir = get_save_dir()
    checkpoint_dir = os.path.join(save_dir, "checkpoints")  # 체크포인트용 디렉토리 추가
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)  # 체크포인트 디렉토리 생성
    print(f"Saving models and logs to: {save_dir}")
    print(f"Saving periodic checkpoints to: {checkpoint_dir}")

    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setup logging
    log_path = os.path.join(save_dir, f"{model_name}.json")
    experiment_data = read_json_file(log_path)

    if use_checkpoint:
        num_of_epochs_already_performed = len(experiment_data)
        checkpoint_path = os.path.join(save_dir, f"{model_name}.pt")
        if os.path.exists(checkpoint_path):
            model = torch.load(checkpoint_path)
    else:
        num_of_epochs_already_performed = 0

    if num_of_epochs_already_performed != 0:
        training_losses = [e['training_loss'] for e in experiment_data]
        training_mious = [e['training_miou'] for e in experiment_data]
        validation_losses = [e['val_loss'] for e in experiment_data]
        validation_mious = [e['val_miou'] for e in experiment_data]
        best_val_performance = min(validation_losses)
    else:
        training_losses = []
        training_mious = []
        validation_losses = []
        validation_mious = []
        best_val_performance = float('inf')

    print(f"Continuing training from Epoch {num_of_epochs_already_performed}")

    model.train()

    for epoch in range(num_of_epochs_already_performed, num_epochs):
        epoch_start_time = time.time()
        training_loss = 0.
        training_miou = 0.
        batch_count = len(train_loader)

        # Training loop
        for i, (image, point_cloud) in enumerate(train_loader):
            batch_start_time = time.time()

            image, point_cloud = Variable(image), Variable(point_cloud)
            image = image.float().to(device=gpu_or_cpu)
            point_cloud = point_cloud.float().to(device=gpu_or_cpu)

            pred = model(image)

            # Calculate loss
            dist = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist[0]) + torch.mean(dist[1])) / 100.0  # Normalize

            miou = calculate_3d_miou(pred, point_cloud)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_miou += miou.item()

            # Progress output
            elapsed_time = time.time() - epoch_start_time
            time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            print(f"Epoch: [{epoch+1}/{num_epochs}] | "
                  f"Batch: [{i+1}/{batch_count}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"mIoU: {miou.item():.4f} | "
                  f"Time: {time_str}")

        avg_training_loss = training_loss / len(train_loader)
        avg_training_miou = training_miou / len(train_loader)

        val_loss, val_miou = val(model, chamferDist, val_loader)

        training_losses.append(avg_training_loss)
        training_mious.append(avg_training_miou)
        validation_losses.append(val_loss)
        validation_mious.append(val_miou)

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training - Loss: {avg_training_loss:.4f}, mIoU: {avg_training_miou:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")
        print(f"Time: {str(datetime.timedelta(seconds=int(epoch_time)))}\n")

        # Save checkpoints
        # Best model 저장
        if val_loss < best_val_performance:
            best_val_performance = val_loss
            best_model_path = os.path.join(save_dir, f"best-{model_name}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'miou': val_miou
            }, best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")

        # 현재 모델 저장
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'miou': val_miou
        }, model_path)

        # 5에폭마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'miou': val_miou,
                'training_losses': training_losses,
                'training_mious': training_mious,
                'validation_losses': validation_losses,
                'validation_mious': validation_mious
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

        # Save logs
        epoch_data = {
            'training_loss': avg_training_loss,
            'training_miou': avg_training_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'epoch': epoch + 1,
            'time': str(datetime.timedelta(seconds=int(epoch_time)))
        }
        experiment_data.append(epoch_data)
        write_json_to_file(log_path, experiment_data)

    return training_losses, validation_losses, torch.load(os.path.join(save_dir, f"best-{model_name}.pt"))

def val(model, chamferDist, val_loader):
    model.eval()
    total_val_loss = 0
    total_val_miou = 0

    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    with torch.no_grad():
        for i, (image, point_cloud) in enumerate(val_loader):
            image, point_cloud = Variable(image), Variable(point_cloud)
            image = image.float().to(device=gpu_or_cpu)
            point_cloud = point_cloud.float().to(device=gpu_or_cpu)

            pred = model(image)

            dist = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist[0]) + torch.mean(dist[1])) / 100.0

            miou = calculate_3d_miou(pred, point_cloud)

            total_val_loss += loss.item()
            total_val_miou += miou.item()

    model.train()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_miou = total_val_miou / len(val_loader)

    return avg_val_loss, avg_val_miou

def write_json_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)

def read_json_file(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        return []

"""
