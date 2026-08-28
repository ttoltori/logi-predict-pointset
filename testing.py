# =============================================================================
# testing.py — A-Point-Set-Generation 프로젝트의 시각화 테스트 스크립트
# -----------------------------------------------------------------------------
# 이 스크립트는 학습이 완료된 모델(.pt)을 로드한 뒤, 테스트 이미지를 입력하여
# 3D 포인트 클라우드를 예측하고, 그 결과를 3D scatter plot으로 시각화하여
# PNG 파일로 저장합니다.
#
# 왜 이 스크립트가 필요한가?
#   학습된 모델이 실제로 어떤 3D 형태를 복원하는지 사람의 눈으로 직접 확인하려면
#   숫자(loss, IoU)만으로는 부족합니다. 예측된 점 구름을 3D 그래프로 그려보면
#   모델이 박스의 형태를 얼마나 잘 복원했는지 직관적으로 파악할 수 있습니다.
# =============================================================================

# Python 2/3 호환성을 위한 print 함수 사용 (레거시 코드 호환 목적)
from __future__ import print_function

# ── 표준 라이브러리 임포트 ──────────────────────────────────────────────────
import argparse   # 명령행 인자 파싱 (실행 시점에 경로/설정을 유연하게 변경하기 위해 사용)
import os         # 파일 경로 조작 및 디렉토리 생성
import random     # 난수 생성 (재현성 확보용 시드 설정 등)
import numpy as np  # 수치 계산 (포인트 클라우드 배열 처리)

# ── PyTorch 및 관련 모듈 임포트 ──────────────────────────────────────────────
# 왜 PyTorch를 사용하는가: 딥러닝 모델의 학습/추론을 효율적으로 수행할 수 있는
# 프레임워크이며, GPU 가산을 통한 빠른 행렬 연산과 자동 미분을 지원합니다.
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn   # cuDNN( NVIDIA GPU 딥러닝 가속 라이브러리) 설정
import torch.optim as optim             # 옵티마이저 (이 스크립트는 추론 전용이므로 실제로 미사용)
import torch.utils.data
import torchvision.datasets as dset     # torchvision 데이터셋 (참고용, 실제 커스텀 로더 사용)
import torchvision.transforms as transforms  # 이미지 전처리 파이프라인 (Resize, ToTensor 등)
import torchvision.utils as vutils      # 이미지 유틸리티 (그리드 생성 등, 참고용)
from torch.autograd import Variable     # 과거 버전 호환용 변수 래퍼 (현재는 tensor만으로 충분)

# ── 프로젝트 내부 모듈 임포트 ────────────────────────────────────────────────
# datasets.py의 PartDataset — 원본 ShapeNet용 데이터셋 클래스 (현재 프로젝트에서는 미사용)
from datasets import PartDataset
import torch.nn.functional as F         # 함수형 신경망 연산 (ReLU, pooling 등)
from torch.cuda import cuda             # CUDA 관련 유틸 (GPU 확인용)

# pic2points: SqueezeNet 백본 + 포인트 생성 헤드로 구성된 핵심 모델 클래스
from pic2points_model import pic2points
from torch.nn.parallel import DataParallel  # 다중 GPU 병렬 처리 (이 스크립트에서는 단일 GPU/CPU 사용)
from torch.autograd import Variable         # 중복 임포트 (레거시 호환성)

# ChamferDistance: 예측 점과 정답 점 사이의 양방향 최근접 거리를 계산하는 손실 함수
# 이 스크립트에서는 시각화 전용이므로 실제 loss 계산에 사용되지는 않지만,
# 향후 확장을 위해 임포트되어 있음
from chamfer_distance import ChamferDistance

# XDataset: 이미지-포인트클라우드 쌍을 로드하는 커스텀 Dataset 클래스
# get_loader: DataLoader를 생성하는 헬퍼 함수 (배치/셔플/워커 설정 일괄 처리)
from data_loader import XDataset, get_loader

# split_data: train/val/test 분할 및 파일명 리스트 읽기/쓰기 유틸
from split_data import read_from_file, write_to_file, split_data

import time  # 실행 시간 측정용 (참고용)

# train: 학습 로직 함수 (이 스크립트는 추론 전용이므로 미사용, 향후 통합용 임포트)
from train_ajit import train

# Visualize: 3D 포인트 클라우드를 matplotlib scatter plot으로 시각화하는 클래스
from visualize import Visualize


def get_next_visual_dir(base_path):
    """
    시각화 결과를 저장할 디렉토리 경로를 결정하는 헬퍼 함수.

    왜 이 함수가 필요한가?
        매번 시각화를 실행할 때마다 같은 디렉토리에 저장하면 이전 결과가 덮어씌워져
        비교가 불가능합니다. 따라서 기본 경로가 이미 존재하면 _1, _2, _3 ... 과 같이
        번호를 붙여 새 디렉토리를 생성함으로써, 실행 마다의 결과를 보존하고
        모델 개선 과정을 추적할 수 있도록 합니다.
    """
    # 기본 경로가 아직 존재하지 않으면 그대로 사용 (첫 실행 또는 이전 결과 삭제 후)
    if not os.path.exists(base_path):
        return base_path

    # 기본 경로가 이미 존재하면 _1, _2, ... 번호를 순차적으로 증가시키며
    # 비어있는(존재하지 않는) 경로를 찾을 때까지 반복
    counter = 1
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def main():
    """
    시각화 테스트 메인 함수.

    전체 흐름:
        1. 명령행 인자 파싱 (경로, 워커 수, 시각화 포인트 수 등)
        2. GPU/CPU 디바이스 설정
        3. 테스트 데이터 로더 생성
        4. 학습된 모델 로드 및 eval 모드 전환
        5. 각 테스트 이미지에 대해 포인트 클라우드 예측
        6. 예측 결과를 3D scatter plot으로 시각화하여 PNG 저장
    """

    # ── 명령행 인자 파싱 ──────────────────────────────────────────
    # 왜 argparse를 쓰는가: 하드코딩된 경로를 매번 코드를 수정하지 않고
    # 실행 시점에 지정할 수 있어, Windows/Linux 어디서든 유연하게 실행 가능
    parser = argparse.ArgumentParser(
        description='A-Point-Set-Generation 시각화 테스트 스크립트')
    parser.add_argument('--image_root', default='datasets/images_poly_bbox_crop', type=str,
                        help='이미지 데이터 루트 경로 (상대/절대 경로 모두 가능)')
    parser.add_argument('--point_cloud_root', default='datasets/labels', type=str,
                        help='포인트 클라우드 데이터 루트 경로 (상대/절대 경로 모두 가능)')
    parser.add_argument('--model_path', default='best-Baseline_DL_Vis.pt', type=str,
                        help='학습된 모델 체크포인트 경로')
    parser.add_argument('--test_list', default='datasets/test_data.txt', type=str,
                        help='테스트 데이터 파일명 리스트 (기본: test_data.txt)')
    parser.add_argument('--save_dir', default='./result/visual', type=str,
                        help='시각화 결과 저장 디렉토리 (기본: ./result/visual)')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='DataLoader 워커 수 (기본: 8, Windows에서는 0 권장)')
    parser.add_argument('--visualize_points', default=3000, type=int,
                        help='시각화에 사용할 포인트 수 (기본: 3000)')
    args = parser.parse_args()

    # 경로를 절대 경로로 변환 (상대 경로도 처리 가능)
    # 왜 절대 경로로 변환하는가: 작업 디렉토리에 따라 상대 경로 해석이 달라지는
    # 문제를 방지하고, 로그 출력 시 사용자가 실제 경로를 명확히 확인할 수 있도록 함
    image_root = os.path.abspath(args.image_root)
    point_cloud_root = os.path.abspath(args.point_cloud_root)
    model_path = os.path.abspath(args.model_path)

    # GPU 설정
    # 왜 GPU를 우선 사용하는가: 딥러닝 추론은 대량의 행렬 연산을 포함하므로
    # GPU가 CPU보다 수십~수백 배 빠릅니다. GPU가 없으면 CPU로 폴백(fallback)합니다.
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
        # 왜 empty_cache를 호출하는가: 이전 실행에서 남은 GPU 메모리 캐시를 비워
        # 메모리 부족(Out of Memory) 오류를 예방하기 위함
        torch.cuda.empty_cache()
    else:
        gpu_or_cpu = torch.device('cpu')

    # ChamferDistance 객체 생성 (이 스크립트에서는 시각화 전용이라 직접 사용하지 않으나,
    # 향후 loss 비교 등을 위해 미리 생성해 둠)
    chamferDist = ChamferDistance()

    # 설정값 출력 — 사용자가 경로가 올바르게 인식되었는지 확인할 수 있도록 함
    print(f'image_root = {image_root}')
    print(f'point_cloud_root = {point_cloud_root}')
    print(f'model_path = {model_path}')

    # ── 추론 하이퍼파라미터 설정 ─────────────────────────────────────────────
    # 왜 batch_size=1인가: 시각화는 각 이미지별로 개별적으로 처리하여 별도의 PNG로
    # 저장해야 하므로, 배치 단위가 아닌 한 장씩 처리하는 것이 결과 관리에 용이합니다.
    batch_size = 1
    # 왜 shuffle=False인가: 테스트 순서를 파일명 리스트 순서대로 유지하여
    # 결과 파일과 원본 데이터의 대응 관계를 명확히 하기 위함입니다.
    shuffle = False
    num_workers = args.num_workers
    # use_2048: 데이터 로더 내부에서 포인트 클라우드를 2048개로 서브샘플링할지 여부
    # (학습 시 메모리 절약을 위한 옵션, 시각화에서는 예측 결과가 11000개이므로 영향 제한적)
    use_2048 = True
    # img_size=227: SqueezeNet 1.1 아키텍처가 요구하는 고정 입력 크기
    # 왜 227인가: SqueezeNet의 컨볼루션/풀링 레이어 구조상 227×227 입력이
    # 최종 feature map 크기를 13×13로 만들도록 설계되어 있어 변경할 수 없습니다.
    img_size = 227
    # visualize_points: 전체 11000개 점을 모두 그리면 렌더링이 느리고
    # 점이 너무 빽빽하여 형태 파악이 어려우므로, 기본 3000개로 줄여서 시각화
    visualize_points = args.visualize_points

    # 이미지 전처리 파이프라인 구성
    # 왜 이 순서인가: 먼저 Resize로 이미지를 227×227에 맞추고, CenterCrop으로
    # 정확히 227×227 크기를 보장한 뒤, ToTensor로 [0,1] 범위의 텐서로 변환합니다.
    # ImageNet 표준 정규화(mean/std)는 원본 코드 설계상 적용하지 않습니다.
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    # 테스트 데이터 로드
    # test_data.txt에서 테스트용 파일명 리스트를 읽어옵니다.
    test_data = read_from_file(args.test_list)
    # get_loader로 DataLoader 생성 — 이미지-포인트클라우드 쌍을 배치 단위로 반환
    test_data_loader = get_loader(image_root, point_cloud_root, test_data, use_2048,
                                transform, batch_size, shuffle, num_workers)

    # 모델 로드
    # 왜 map_location을 쓰는가: 모델이 GPU에서 학습되었더라도, CPU 환경에서
    # 로드할 수 있도록 텐서를 지정한 디바이스로 매핑하기 위함입니다.
    model = torch.load(model_path, map_location=gpu_or_cpu, weights_only=False)
    # 모델을 해당 디바이스(GPU 또는 CPU)로 이동
    model = model.to(device=gpu_or_cpu)
    # 왜 eval()을 호출하는가: 추론 모드로 전환하여 Dropout을 비활성화하고
    # BatchNorm을 학습 시의 통계값을 사용하는 추론 통계로 고정합니다.
    model.eval()

    # 시각화 결과 저장 폴더 경로 설정
    # get_next_visual_dir로 기존 디렉토리와 충돌하지 않는 새 경로를 결정
    save_dir = get_next_visual_dir(args.save_dir)
    # exist_ok=True: 이미 존재해도 에러를 발생시키지 않음 (안전장치)
    os.makedirs(save_dir, exist_ok=True)
    print(f"결과 저장 경로: {save_dir}")

    # 왜 torch.no_grad()를 사용하는가: 추론 시에는 기울기(gradient)를 계산할
    # 필요가 없으므로, 메모리 사용량을 줄이고 속도를 높이기 위해 자동 미분을 끕니다.
    with torch.no_grad():
        # 테스트 데이터 로더에서 이미지와 정답 포인트 클라우드를 한 장씩 가져옴
        for i, (image, point_cloud) in enumerate(test_data_loader):
            # 파일명 처리 — 결과 PNG 파일명을 원본 이미지 파일명과 동일하게 유지하여
            # 어떤 이미지에 대한 결과인지 쉽게 식별할 수 있도록 함
            filename = test_data[i]
            basename = os.path.splitext(filename)[0]  # 확장자 제거 (예: "item.png" → "item")
            original_name = basename

            # Variable로 감싸는 것은 과거 PyTorch 호환성 코드 (현재는 tensor만으로 충분)
            image, point_cloud = Variable(image), Variable(point_cloud)
            # 이미지를 float 텐서로 변환하고 지정된 디바이스로 이동
            image = image.float().to(device=gpu_or_cpu)
            # 정답 포인트 클라우드도 동일하게 처리 (시각화에서는 직접 사용하지 않으나
            # 향후 비교 시각화를 위해 로드함)
            point_cloud = point_cloud.float().to(device=gpu_or_cpu)

            # 모델에 이미지를 입력하여 예측 포인트 클라우드 생성
            # pred의 shape: (batch_size=1, num_points=11000, 3) — 각 점의 (x, y, z) 좌표
            pred = model(image)
            # 시각화를 위해 예측 결과를 CPU로 이동 (matplotlib은 CPU 텐서만 처리)
            pred = pred.to('cpu')

            # 포인트 클라우드 준비
            # detach(): 계산 그래프에서 분리하여 메모리 해제 (추론 후 불필요한 그래프 유지 방지)
            # numpy(): PyTorch 텐서를 NumPy 배열로 변환 (matplotlib/시각화용)
            p_np = pred[0].detach().numpy()
            # 전체 11000개 점을 모두 그리면 느리고 점이 너무 빽빽하여 형태 파악이 어려움
            # 따라서 visualize_points(기본 3000)개로 무작위 샘플링하여 시각화 성능과 가독성 확보
            if len(p_np) > visualize_points:
                # np.random.choice: 중복 없이(replace=False) 무작위로 인덱스 선택
                indices = np.random.choice(len(p_np), visualize_points, replace=False)
                p_np = p_np[indices]

            # 시각화 및 저장 - 이미지 텐서도 전달
            # Visualize 클래스에 샘플링된 포인트 클라우드를 전달하여
            # 원본 이미지와 3D scatter plot을 나란히 배치한 PNG를 저장
            Visualize(p_np).ShowResult(save_path=os.path.join(save_dir, original_name), img_tensor=image[0])
            # 진행 상황 출력 (현재/전체) — 대량 데이터 시 얼마나 진행되었는지 파악용
            print(f'처리완료 {i+1}/{len(test_data_loader)}: {original_name}')

    # =========================================================================
    # 아래는 이전 버전의 시각화 루프 (주석 처리됨)
    # 왜 유지하는가: test_data가 (image_type, specific_type) 튜플 형태였던 과거
    # 데이터 로더 구조를 사용하던 코드로, 현재 단일 파일명 기반 로더로 변경되어
    # 비활성화되었지만 참고용으로 보존합니다.
    # =========================================================================
    # with torch.no_grad():
    #     for i, (image, point_cloud) in enumerate(test_data_loader):
    #         # 원본 이미지 이름 가져오기
    #         image_type, specific_type = test_data[i]
    #         original_name = f"{image_type}_{specific_type.split('.')[0]}"
            
    #         image, point_cloud = Variable(image), Variable(point_cloud)
    #         image = image.float().to(device=gpu_or_cpu)
    #         point_cloud = point_cloud.float().to(device=gpu_or_cpu)
            
    #         pred = model(image)
    #         pred = pred.to('cpu')
            
    #         # 포인트 클라우드 준비
    #         p_np = pred[0].detach().numpy()
    #         if len(p_np) > visualize_points:
    #             indices = np.random.choice(len(p_np), visualize_points, replace=False)
    #             p_np = p_np[indices]
            
    #         # 시각화 및 저장
    #         Visualize(p_np).ShowResult(
    #             img_tensor=image.cpu(),
    #             save_path=os.path.join(save_dir, original_name)
    #         )
    #         print(f'처리완료 {i+1}/{len(test_data_loader)}: {original_name}')


if __name__ == "__main__":
    # 스크립트가 직접 실행될 때만 main() 호출 (모듈 임포트 시에는 실행되지 않음)
    main()

"""
=============================================================================
아래는 이전 버전의 main() 함수입니다 (삼중 따옴표 문자열로 보존됨).
왜 유지하는가: batch_size=64, Chamfer Distance loss 계산, ShowRandom() 사용 등
과거 접근 방식을 참고하기 위해 그대로 보존합니다. 현재 활성 코드는 위의
argparse 기반 버전입니다.
=============================================================================
def main():
    chamferDist = ChamferDistance()
    
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    image_root = "/workspace/DATA/NIA29_3D/images_poly_bbox_crop"
    point_cloud_root = "/workspace/DATA/NIA29_3D/labels"

    batch_size = 64
    shuffle = True
    num_workers = 8
    use_2048 = True
    img_size = 227
    num_points = 2000  # 시각화할 포인트 수
    transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                    transforms.CenterCrop(img_size),transforms.ToTensor()])
    
    path_test = 'test_data.txt'
    test_data = read_from_file(path_test)
    
    test_data_loader = get_loader(image_root, point_cloud_root, test_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)

    model = torch.load('/workspace/MODELS/A-Point-Set-Generation-Network-for-3D-Object-Reconstruction-from-a-Single-Image/best-Baseline_DL_Vis.pt').to(device=gpu_or_cpu)
    model.eval()

    for i, (image, point_cloud) in enumerate(test_data_loader):
        image, point_cloud = Variable(image, requires_grad = False), Variable(point_cloud, requires_grad = False)
        image, point_cloud = image.float().to(device=gpu_or_cpu), point_cloud.float().to(device=gpu_or_cpu)
        pred = model(image)
        dist = chamferDist(pred, point_cloud)
        loss = torch.mean(dist[0]) + torch.mean(dist[1])
        
        print('pred size = ', pred.size())
        pred = pred.to('cpu')
        
        out = []
        
        for p in pred:
            # Convert to numpy and randomly sample num_points
            p_np = p.detach().numpy()
            if len(p_np) > num_points:
                # Randomly sample num_points if we have more points than needed
                indices = np.random.choice(len(p_np), num_points, replace=False)
                p_np = p_np[indices]
            out.append(p_np)
            
        print('type(out) = ', type(out))
        print(f'Number of points per cloud = {len(out[0])}')  # Should print num_points
        
        print('Visualize the prediction')
        Visualize(out).ShowRandom()
        break

if __name__ == "__main__":
    main()
"""
