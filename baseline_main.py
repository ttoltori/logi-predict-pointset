# ===========================================================================
# baseline_main.py
# A-Point-Set-Generation 프로젝트의 학습/테스트 엔트리포인트
# ---------------------------------------------------------------------------
# 이 파일은 단일 상품 이미지로부터 3D 포인트 클라우드(11,000개 점)를
# 생성하는 모델을 학습하고, Pix3D 테스트 데이터로 Chamfer Distance를
# 평가하는 전체 파이프라인을 담당합니다.
# ===========================================================================

# Python 2/3 호환성을 위한 print 함수 사용 (구버전 환경에서도 동작 보장)
from __future__ import print_function

# --- 표준 라이브러리 임포트 ---
# argparse: 명령행 인자를 파싱하여 실행 시점에 경로/설정을 유연하게 변경
import argparse
# os: 파일 경로 조작 및 절대 경로 변환을 위해 사용
import os
# random: 데이터 셔플 등 무작위 처리용
import random
# numpy: 포인트 클라우드 배열 연산용
import numpy as np

# --- PyTorch 핵심 임포트 ---
# torch: 텐서 연산 및 딥러닝 프레임워크 핵심
import torch
# nn: 신경망 모듈(레이어, 손실함수 등) 정의용
import torch.nn as nn
# nn.parallel: 내부 API 참조용 (DataParallel과 함께 다중 GPU 학습 지원)
import torch.nn.parallel
# cudnn: GPU 합성곱 연산 성능 최적화(벤치마크 모드 활성화용)
import torch.backends.cudnn as cudnn
# optim: 옵티마이저(Adam 등) 정의용
import torch.optim as optim
# utils.data: DataLoader로 배치 단위 데이터 로딩
import torch.utils.data
# torchvision.datasets: 이미지 데이터셋 로딩 (현재는 커스텀 로더 사용)
import torchvision.datasets as dset
# torchvision.transforms: 이미지 전처리(Resize, CenterCrop, ToTensor 등)
import torchvision.transforms as transforms
# torchvision.utils: 이미지 저장/시각화 유틸
import torchvision.utils as vutils
# Variable: 과거 PyTorch 버전 호환용 (현재는 tensor만으로 충분하지만 레거시 유지)
from torch.autograd import Variable
# PartDataset: 원본 ShapeNet용 데이터셋 (현재 NIA29 커스텀 로더 사용으로 미사용)
from datasets import PartDataset
# F: 함수형 API (ReLU, 풀링 등)
import torch.nn.functional as F
# cuda: GPU 관련 유틸 (현재는 torch.cuda로 직접 접근)
import torch.cuda as cuda

# --- 프로젝트 내부 모듈 임포트 ---
# pic2points: SqueezeNet 백본 + 포인트 생성 헤드로 구성된 핵심 모델 클래스
from pic2points_model import pic2points
# DataParallel: 다중 GPU 환경에서 모델을 병렬로 분산 학습시키기 위한 래퍼
from torch.nn.parallel import DataParallel
# Variable: autograd 변수 (레거시 호환용 중복 임포트)
from torch.autograd import Variable
import torch
# ChamferDistance: 예측 점과 정답 점 사이의 양방향 최근접 거리를 계산하는 손실 함수
from chamfer_distance import ChamferDistance
# XDataset, get_loader: 학습/검증용 커스텀 데이터셋 및 DataLoader 생성 함수
from data_loader import XDataset, get_loader
# read_from_file, write_to_file, split_data: 데이터 분할 및 파일명 리스트 읽기/쓰기 유틸
from split_data import read_from_file, write_to_file, split_data
# time: 학습 소요 시간 측정용
import time
# train: train_ajit.py의 학습 루프 함수 (체크포인트 저장, 검증, 로그 기록 포함)
from train_ajit import train
# TestDataset: Pix3D 테스트용 데이터셋 (객체 ID 기반으로 이미지-포인트클라우드 쌍 로딩)
from data_loader_pix3d import TestDataset
# argparse, sys: 명령행 인자 및 시스템 유틸 (중복 임포트)
import argparse, sys


def main():
    # ===========================================================================
    # main() 함수
    # 학습 및 테스트의 전체 흐름을 조율하는 엔트리포인트 함수입니다.
    # argparse로 설정을 받아 데이터 로더 생성 → 모델 생성 → 학습/로드 →
    # Pix3D 테스트 평가 순으로 실행합니다.
    # ===========================================================================

    # GPU 설정
    # cudnn.benchmark=True: 입력 크기가 고정일 때 cuDNN이 최적의 합성곱 알고리즘을
    # 자동 탐색하여 성능을 향상시킵니다. 단, 입력 크기가 매번 변하면 오히려
    # 오버헤드가 발생할 수 있으나 본 프로젝트는 227×227 고정이므로 안전합니다.
    torch.backends.cudnn.benchmark = True
    # GPU가 사용 가능한 경우 캐시를 비워 메모리 단편화를 줄입니다.
    # 학습 시작 전 남아있는 이전 세션의 메모리를 정리하여 OOM을 방지합니다.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 명령행 인자 파싱 ──────────────────────────────────────────
    # 왜 argparse를 쓰는가: 하드코딩된 경로를 매번 코드를 수정하지 않고
    # 실행 시점에 지정할 수 있어, Windows/Linux 어디서든 유연하게 실행 가능
    parser = argparse.ArgumentParser(
        description='A-Point-Set-Generation 학습/테스트 엔트리포인트')

    # --training: 학습 모드 여부. False로 설정하면 기존 체크포인트를 로드하여
    # 테스트만 수행합니다. 문자열로 받아 아래에서 boolean으로 변환합니다.
    parser.add_argument('--training', default='True', type=str,
                        help='학습 모드 여부 (True/False, 기본: True)')

    # --image_root: 전처리된 상품 이미지(PNG)가 저장된 디렉토리.
    # prepare_data_3d.py가 생성한 datasets/images_poly_bbox_crop 가 기본값.
    parser.add_argument('--image_root', default='datasets/images_poly_bbox_crop', type=str,
                        help='이미지 데이터 루트 경로 (상대/절대 경로 모두 가능)')

    # --point_cloud_root: 정답 포인트 클라우드(NPY)가 저장된 디렉토리.
    # 모델이 학습해야 할 정답(Ground Truth) 3D 좌표 데이터의 위치입니다.
    parser.add_argument('--point_cloud_root', default='datasets/labels', type=str,
                        help='포인트 클라우드 데이터 루트 경로 (상대/절대 경로 모두 가능)')

    # --train_list / --val_list / --test_list: split_data.py가 생성한
    # 파일명 리스트 파일. 각 파일에는 학습/검증/테스트에 사용할 이미지 파일명이
    # 한 줄씩 기록되어 있어, 전체 데이터를 8:1:1 비율로 분할하여 사용합니다.
    parser.add_argument('--train_list', default='datasets/train_data.txt', type=str,
                        help='학습 데이터 파일명 리스트 (기본: train_data.txt)')
    parser.add_argument('--val_list', default='datasets/val_data.txt', type=str,
                        help='검증 데이터 파일명 리스트 (기본: val_data.txt)')
    parser.add_argument('--test_list', default='datasets/test_data.txt', type=str,
                        help='테스트 데이터 파일명 리스트 (기본: test_data.txt)')

    # --model_name: 체크포인트 파일명과 학습 로그 파일명에 사용되는 식별자.
    # 예: best-Baseline_DL_Vis.pt, Baseline_DL_Vis.pt, Baseline_DL_Vis(JSON 로그)
    parser.add_argument('--model_name', default='Baseline_DL_Vis', type=str,
                        help='모델 저장 이름 (체크포인트 파일명에 사용)')

    # --num_epochs: 전체 학습 데이터를 몇 번 반복 학습할지 지정.
    # 너무 적으면 과소적합, 너무 많으면 과대적합 및 시간 낭비가 발생합니다.
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='학습 epoch 수 (기본: 50)')

    # --batch_size: 한 번에 처리할 이미지 수. GPU 메모리에 따라 조절 필요.
    # Windows 환경이나 메모리가 작은 GPU에서는 4~8 정도로 줄이는 것이 안전합니다.
    parser.add_argument('--batch_size', default=32, type=int,
                        help='배치 크기 (기본: 32, Windows/GPU 메모리에 따라 축소)')

    # --num_workers: DataLoader가 데이터를 병렬로 로딩할 프로세스 수.
    # Windows에서는 멀티프로세싱 이슈가 있어 0(메인 프로세스만 사용)을 권장합니다.
    parser.add_argument('--num_workers', default=8, type=int,
                        help='DataLoader 워커 수 (기본: 8, Windows에서는 0 권장)')

    # --learning_rate: 옵티마이저(Adam)의 학습률.
    # 너무 크면 발산, 너무 작으면 수렴이 느려집니다. 0.001은 Adam의 표준값입니다.
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='학습률 (기본: 0.001)')

    # --num_points: 모델이 출력할 3D 점의 개수. NIA29 데이터는 11,000개 점 사용.
    # 점이 많을수록 형태 재구성이 정밀해지지만 연산량과 메모리가 증가합니다.
    parser.add_argument('--num_points', default=11000, type=int,
                        help='포인트 클라우드 점 수 (기본: 11000)')

    # --use_checkpoint: 이전에 저장된 체크포인트에서 학습을 이어서 진행할지 여부.
    # 학습 중단 후 재개할 때 유용하며, train_ajit.py의 JSON 로그에서 진행된 epoch를 읽습니다.
    parser.add_argument('--use_checkpoint', default=False, action='store_true',
                        help='기존 체크포인트에서 학습 재개')

    # --cpu: GPU 대신 CPU로 강제 학습.
    # 왜 필요한가: GPU 메모리가 부족한 경우(8GB GPU에서 batch_size=32 + 11000점 포인트 클라우드) OOM이 발생.
    # CPU는 느리지만 메모리 제한이 없어 디버깅이나 소규모 테스트에 적합하다.
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='GPU 대신 CPU로 강제 학습 (메모리 부족 시 사용, 속도 느림)')
    args = parser.parse_args()

    # 경로를 절대 경로로 변환 (상대 경로도 처리 가능)
    # 왜 변환하는가: os.walk 등이 실행 디렉토리에 의존하지 않도록 보장
    # 사용자가 어느 디렉토리에서 스크립트를 실행하더라도 데이터 경로가
    # 일관되게 해석되도록 절대 경로로 정규화합니다.
    image_root = os.path.abspath(args.image_root)
    point_cloud_root = os.path.abspath(args.point_cloud_root)

    # training 인자를 문자열에서 boolean으로 변환합니다.
    # argparse가 기본적으로 문자열로 받기 때문에, 'True'/'true' 외의 값은
    # 모두 False로 처리하여 테스트 전용 모드로 진입할 수 있게 합니다.
    is_training = args.training

    if (is_training is None) or (is_training == 'True') or (is_training == 'true'):
        is_training = True
    else:
        is_training = False
    # 현재 모드와 경로를 출력하여 실행 전 설정이 올바른지 사용자가 확인할 수 있도록 합니다.
    print('is_training  mode = ', is_training)
    print(f'image_root = {image_root}')
    print(f'point_cloud_root = {point_cloud_root}')

    # ChamferDistance 객체를 미리 생성합니다.
    # 이 손실 함수는 예측 점 집합과 정답 점 집합 사이의 양방향 최근접 거리를
    # 계산하여, 두 3D 형태가 얼마나 유사한지를 수치화합니다.
    # 학습(train_ajit.py)과 테스트(아래 루프) 모두에서 동일한 객체를 재사용합니다.
    chamferDist = ChamferDistance()

    # Decide on GPU or CPU
    # CUDA(GPU)가 사용 가능한 경우 GPU를, 그렇지 않은 경우 CPU를 사용하도록
    # 디바이스를 결정합니다. 이 분기 처리 덕분에 GPU가 없는 환경(예: 일반 노트북)에서도
    # 코드가 실행 가능합니다. 단, CPU 환경에서는 학습 속도가 매우 느립니다.
    # --cpu 플래그가 지정된 경우, GPU가 있어도 CPU를 강제 사용합니다.
    # 왜 강제 CPU가 필요한가: GPU 메모리(8GB)가 부족하여 OOM이 발생하는 경우,
    # CPU는 메모리 제한이 없으므로 느리더라도 학습을 완료할 수 있습니다.
    if args.cpu:
        gpu_or_cpu = torch.device('cpu')
        print('CPU 강제 모드 (--cpu)')
    elif torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    # Training Configuration
    # 학습에 사용할 하이퍼파라미터들을 지역 변수로 설정합니다.
    # argparse에서 받은 값을 그대로 사용하되, 일부는 고정값을 사용합니다.
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    # shuffle=True: 매 epoch마다 데이터 순서를 섞어 모델이 데이터 순서에
    # 의존적으로 학습되는 것(일반화 성능 저하)을 방지합니다.
    shuffle = True
    num_workers = args.num_workers
    # use_2048: 데이터 로더에 2048개 점을 사용할지 여부를 전달하는 플래그.
    # 학습 시에는 2048개 점을 샘플링하여 사용합니다(메모리/속도 효율).
    use_2048 = True
    # img_size=227: SqueezeNet 1.1 백본의 입력 크기.
    # 원본 SqueezeNet이 227×227로 학습되었기 때문에 이 크기를 유지해야
    # 사전학습 가중치를 그대로 활용할 수 있습니다. 224가 아닌 227인 이유는
    # 원본 구현의 설계 선택입니다.
    img_size = 227 # I don't know why, but this has to be 227!
    learning_rate = args.learning_rate
    num_points = args.num_points

    # 이미지 전처리 파이프라인을 정의합니다.
    # Resize → CenterCrop → ToTensor 순서로 처리합니다.
    # - Resize: 이미지를 227×227로 확대/축소 (interpolation=2는 양선형 보간)
    # - CenterCrop: 중앙을 기준으로 227×227 크롭 (여기서는 Resize와 동일 크기라 변화 없음)
    # - ToTensor: PIL 이미지(0~255)를 텐서(0~1)로 변환 및 채널 우선 형태로 변경
    # 주의: ImageNet 표준 정규화(mean/std)를 적용하지 않는 것이 원본 설계 선택입니다.
    transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                    transforms.CenterCrop(img_size),transforms.ToTensor()])

    # Checkpoint
    # use_checkpoint: 학습 재개 여부. True이면 이전 체크포인트와 JSON 로그를
    # 읽어 중단된 epoch부터 이어서 학습합니다.
    use_checkpoint = args.use_checkpoint

    # Split and Get data. Override the saved files if you change the ratios.
    # 데이터를 train/val/test = 8:1:1 비율로 분할합니다.
    # overrideFiles=False: 이미 분할 파일이 존재하면 덮어쓰지 않습니다.
    # 이렇게 하면 분할 결과를 재현 가능하게 유지하여 실험의 일관성을 보장합니다.
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    split_data(train_ratio, val_ratio, test_ratio, overrideFiles = False)

    # 분할된 파일명 리스트 파일에서 실제 파일명들을 읽어옵니다.
    # 각 파일에는 학습/검증/테스트에 사용할 이미지 파일명이 한 줄씩 저장되어 있습니다.
    path_train = args.train_list
    path_val = args.val_list
    path_test = args.test_list

    train_data = read_from_file(path_train)
    val_data = read_from_file(path_val)
    test_data = read_from_file(path_test)

    # Data loader
    # get_loader() 함수로 학습/검증/테스트용 DataLoader를 생성합니다.
    # DataLoader는 이미지와 포인트 클라우드를 배치 단위로 모델에 공급하는 역할을 합니다.
    # 동일한 transform과 use_2048 설정을 모두에 적용하여 전처리 일관성을 유지합니다.
    train_data_loader = get_loader(image_root, point_cloud_root, train_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)

    val_data_loader = get_loader(image_root, point_cloud_root, val_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)
    test_data_loader = get_loader(image_root, point_cloud_root, test_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)
    
    # 학습 데이터 로더의 길이(배치 수)를 출력하여 데이터가 정상적으로 로딩되었는지 확인합니다.
    print('Len of train loader = ', len(train_data_loader))

    # create model
    # pic2points 모델을 생성합니다.
    # 이 모델은 SqueezeNet 1.1 백본으로 이미지 feature를 추출한 뒤,
    # 1×1 합성곱과 FC 레이어를 통해 num_points(11000)개 점의 (x,y,z) 좌표를 회귀합니다.
    print("model building...")
    model = pic2points(num_points=num_points)

    # 다중 GPU가 감지된 경우 DataParallel로 모델을 감싸 데이터를 GPU 간 분산시킵니다.
    # 단일 GPU 환경에서는 DataParallel을 적용하지 않아 불필요한 오버헤드를 피합니다.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    # 모델을 앞서 결정한 디바이스(GPU 또는 CPU)로 이동시킵니다.
    # .to()를 사용하면 디바이스가 무엇이든 동일한 코드로 처리할 수 있어 호환성이 높습니다.
    model.to(device=gpu_or_cpu)
    
    # 학습 모드인 경우 train_ajit.py의 train() 함수를 호출하여 학습을 시작합니다.
    # train()은 내부적으로 Adam 옵티마이저로 Chamfer Distance를 최소화하며,
    # 매 epoch마다 체크포인트를 저장하고 검증 loss가 최소일 때 best 모델을 저장합니다.
    if is_training:
        # Train
        print('Starting training...')
        train_losses, val_loss, best_model = train(model, train_data_loader, val_data_loader, chamferDist,
                                                   model_name=args.model_name, num_epochs=num_epochs,
                                                   lr=learning_rate, use_checkpoint = use_checkpoint,
                                                   device=gpu_or_cpu)
    else:
        # 테스트 전용 모드: 학습하지 않고 기존에 저장된 best 체크포인트를 로드합니다.
        # map_location을 지정하여 GPU에서 저장된 모델도 CPU 환경에서 로드할 수 있도록 합니다.
        best_model = torch.load(f'best-{args.model_name}.pt', map_location=gpu_or_cpu, weights_only=False)
        print(f'Loaded previously saved model: best-{args.model_name}.pt')

    # 왜 .to(gpu_or_cpu)를 쓰는가: CUDA가 없는 환경에서 .cuda() 호출 시 오류 방지
    # 로드한 best 모델을 다시 해당 디바이스로 이동시키고 평가 모드로 전환합니다.
    # model.eval()은 Dropout과 BatchNorm을 추론 모드로 설정하여 평가 결과의
    # 재현성과 정확성을 보장합니다. (학습 시와 동작이 달라지는 레이어를 고정)
    model = best_model.to(gpu_or_cpu)
    model.eval()

    # Compute chamfer distance on Pix3D dataset.
    # 전처리된 데이터 경로를 그대로 사용 (별도 지정하지 않으면 학습/평가와 동일한 경로)
    # 학습에 사용한 데이터와 동일한 이미지 경로를 재사용합니다.
    img_path = image_root
    # 포인트 클라우드는 labels 하위의 npy_stride5 서브디렉토리에 저장되어 있습니다.
    # stride5는 포인트 클라우드 생성 시 5 간격으로 샘플링했음을 의미합니다.
    pc_path = os.path.join(point_cloud_root, 'npy_stride5')

    # Pix3D 테스트에 사용할 객체 ID 리스트입니다.
    # 원본 코드는 박스 크기 코드(B120110053 등)를 하드코딩했지만, NIA29 데이터에서는
    # 파일명이 {KAN4}_{barcode}_{shot}_{cam}.png 형식이므로 test_data.txt에서 실제 파일명을 읽어옵니다.
    # 왜 test_data.txt를 쓰는가: prepare_data_3d.py가 학습/검증/테스트 분할 시 생성한 리스트이며,
    # 실제 전처리된 파일명과 1:1로 매칭되기 때문이다.
    from split_data import read_from_file as _read_test_list
    objects = _read_test_list(args.test_list)
    # 확장자 제거: TestDataset이 filename + ".png" / filename + ".npy" 형식으로 검색하므로
    # 리스트에서 .png 확장자를 제거한 파일명만 전달한다.
    objects = [os.path.splitext(f)[0] if f.endswith('.png') else f for f in objects]

    # TestDataset으로 Pix3D 테스트 데이터셋을 생성합니다.
    # 객체 ID 리스트를 기반으로 이미지와 포인트 클라우드를 매핑하여 로딩합니다.
    test_dataset = TestDataset(img_path,pc_path, objects)

    # 테스트 데이터가 없으면 평가 단계를 건너뛴다.
    # 왜 확인하는가: TestDataset이 빈 리스트를 반환할 수 있으며(파일 매칭 실패 시),
    # 빈 DataLoader에서 shuffle=True로 생성하면 RandomSampler가 0개 샘플 오류를 발생시킨다.
    if len(test_dataset) == 0:
        print('Warning: 테스트 데이터가 없습니다. Pix3D 평가를 건너뜁니다.')
    else:
        # 테스트용 DataLoader를 생성합니다.
        # batch_size=1: 테스트는 정확한 개별 평가를 위해 한 샘플씩 처리합니다.
        # shuffle=True: 테스트 순서를 섞지만, 전체 평균을 구하므로 결과에는 영향이 없습니다.
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=num_workers)

        # Pix3D 데이터셋에 대한 Chamfer Distance 평가를 시작합니다.
        print('Starting testing on Pix3D dataset...')
        # 모든 테스트 샘플의 loss를 누적하여 평균을 구하기 위한 변수입니다.
        total_test_loss = 0.
        # Get loss on training data.
        # torch.no_grad(): 평가 시에는 그래디언트를 계산할 필요가 없으므로
        # 메모리 사용량과 연산 시간을 크게 줄이기 위해 no_grad 컨텍스트를 사용합니다.
        with torch.no_grad():
            # 테스트 데이터 로더에서 이미지와 정답 포인트 클라우드를 한 쌍씩 가져옵니다.
            for i, (image, point_cloud) in enumerate(test_data_loader):

                # Variable로 감싸는 것은 레거시 호환용이며, 최신 PyTorch에서는
                # tensor 자체가 autograd를 지원하므로 실질적인 차이는 없습니다.
                image, point_cloud = Variable(image), Variable(point_cloud)

                # 이미지 채널이 3(RGB)이 아닌 경우(예: 흑백 이미지 등)는 건너뜁니다.
                # 모델이 3채널 입력을 기대하므로, 채널 수가 다르면 오류가 발생하기 때문입니다.
                if (image.size(1) != 3):
                    continue

                # 데이터를 float 타입으로 변환하고 해당 디바이스(GPU/CPU)로 이동시킵니다.
                # 모델과 동일한 디바이스에 있어야 연산이 가능합니다.
                image, point_cloud = image.float().to(device=gpu_or_cpu), point_cloud.float().to(device=gpu_or_cpu)
                # 모델에 이미지를 입력하여 예측 포인트 클라우드를 생성합니다.
                pred = model(image)
                # Chamfer Distance를 계산합니다.
                # dist1: 정답 점 → 예측 점 방향의 최근접 거리 (정답이 예측에 얼마나 잘 덮이는가)
                # dist2: 예측 점 → 정답 점 방향의 최근접 거리 (예측이 정답을 얼마나 잘 덮는가)
                # 두 값을 모두 고려해야 양방향 형태 유사성을 정확히 평가할 수 있습니다.
                dist1, dist2 = chamferDist(pred, point_cloud)
                # 양방향 거리의 평균을 합산하여 loss로 사용합니다.
                # 두 항을 더함으로써 예측이 정답을 놓치지도, 불필요한 점을 만들지도 않도록 유도합니다.
                loss = (torch.mean(dist1)) + (torch.mean(dist2))
                # loss.item()으로 스칼라 값을 추출하여 누적합니다.
                # 텐서를 그대로 더하면 그래디언트 추적이 누적되어 메모리가 증가하므로
                # item()으로 Python float로 변환하는 것이 중요합니다.
                total_test_loss += loss.item()

                # 100 배치마다 진행 상황을 출력하여 긴 테스트 과정을 모니터링할 수 있도록 합니다.
                if i%100 == 0:
                    print('Batch '+str(i)+' finished.')

        # 전체 테스트 loss의 평균을 출력합니다.
        # 이 값이 작을수록 모델이 Pix3D 데이터의 3D 형태를 정확히 재구성함을 의미합니다.
        print('Chamfer distance on Pix3D dataset = ', total_test_loss / len(test_data_loader))
    
# 스크립트가 직접 실행될 때만 main()을 호출합니다.
# 모듈로 임포트되는 경우에는 main()이 자동 실행되지 않도록 보호합니다.
# 이는 다른 스크립트에서 이 파일의 함수나 클래스를 재사용할 수 있게 합니다.
if __name__ == "__main__":
    main()
