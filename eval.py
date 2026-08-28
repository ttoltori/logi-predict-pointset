
# ──────────────────────────────────────────────────────────────────────────
# eval.py — A-Point-Set-Generation 프로젝트 상세 평가 스크립트
#
# 이 스크립트는 학습이 완료된 모델(best 체크포인트)을 불러와 테스트 데이터셋
# 전체에 대해 추론을 수행하고, 각 이미지별 예측 포인트 클라우드를 .npy 파일로
# 저장한 뒤 3D IoU / 바운딩 박스 부피 등의 평가 지표를 계산하여 로그로 남긴다.
#
# 왜 별도의 eval.py가 필요한가?
#   학습 스크립트(baseline_main.py)의 검증(val) 단계는 모델 선택을 위한
#   빠른 손실/IOU 산정만 수행한다. 반면 eval.py는 모든 테스트 샘플에 대해
#   개별 예측 결과를 파일로 저장하고 항목별 지표를 상세히 기록하므로,
#   모델의 최종 성능을 보고서 형태로 남기기 위해 사용한다.
# ──────────────────────────────────────────────────────────────────────────

# 운영체제 경로 처리를 위해 os 모듈 사용 — Windows/Linux 모두에서 경로 구분자
# 문제 없이 동작하도록 os.path.join / os.path.abspath 를 사용한다.
import os

# PyTorch 핵심 — 텐서 연산과 자동 미분(평가 시에는 no_grad로 미분 비활성화)
import torch
# nn.DataParallel 등 다중 GPU 래핑을 위해 필요하다.
import torch.nn as nn
# Variable은 구버전 호환용 — 최신 PyTorch에서는 직접 텐서를 써도 되지만
# 원본 코드가 import 하고 있으므로 유지한다.
from torch.autograd import Variable
# 이미지 전처리(리사이즈, 크롭, 텐서 변환)를 위해 torchvision transforms 사용
import torchvision.transforms as transforms
# 학습/검증용 데이터 로더를 생성하는 get_loader 함수 — Xdataset 기반이다.
from data_loader import get_loader
# split_data의 read_from_file 대신 이 파일 내부에 동일한 함수를 직접 구현하여
# 사용한다(아래 정의). 따라서 아래 import는 주석 처리되어 있다.
#from split_data import read_from_file
# ChamferDistance는 이 평가 스크립트에서는 직접 사용하지 않으나,
# 하단 주석 처리된 이전 버전에서 참조하므로 import 를 유지한다.
from chamfer_distance import ChamferDistance
# calculate_3d_miou 만 단독 import 하는 줄은 주석 처리 — 아래 한 줄에서
# 여러 지표 함수를 한 번에 불러오기 때문에 중복을 피하려는 것이다.
#from metrics import calculate_3d_miou
# 3D IoU(mIoU), 3D 바운딩 박스, 부피 계산 함수 — 평가 지표 산출의 핵심.
from metrics import calculate_3d_miou, calculate_3d_bbox, calculate_volume
# 단일 이미지 → 포인트 클라우드 생성 모델 클래스 (SqueezeNet 백본 + 헤드)
from pic2points_model import pic2points
# 예측 포인트 클라우드를 .npy 파일로 저장하기 위한 numpy
import numpy as np
# 파일/콘솔 동시 로깅을 위한 logging 모듈
import logging
# 로그 파일명에 타임스탬프를 붙이고, 시작 시간을 기록하기 위해 time 사용
import time
# 평가 진행률을 시각적으로 보여주기 위해 tqdm 사용 — 대용량 데이터셋에서
# 남은 시간을 가늠할 수 있어 장시간 실행되는 평가에 필수적이다.
from tqdm import tqdm
# 실행 명령어(sys.argv)를 로그에 기록하여 어떤 파라미터로 실행했는지 추적하기 위함.
import sys


def setup_logger():
    """
    파일과 콘솔에 동시에 로그를 출력하는 로거를 생성하여 반환한다.

    왜 파일과 콘솔 두 군데에 동시에 출력하는가?
        평가는 한 번 실행에 수 분~수십 분이 걸릴 수 있다. 콘솔 출력만으로는
        터미널이 닫히면 결과가 사라지므로 파일에 영구 기록을 남기고, 동시에
        콘솔에서 진행 상황을 실시간으로 확인할 수 있도록 두 핸들러를 모두
        추가한다. 이렇게 하면 사후 분석 시 로그 파일을 재사용할 수 있다.
    """
    # 로그 디렉토리가 없으면 생성 — 평가마다 별도 로그 파일이 생기므로
    # 디렉토리가 항상 존재해야 한다.
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 'evaluation'이라는 이름의 로거를 가져온다. 같은 이름으로 여러 번 호출해도
    # 동일한 로거 인스턴스가 반환되므로 핸들러가 중복 추가되지 않게 주의해야 한다.
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)

    # 파일 핸들러: 로그를 파일에 영구 저장한다.
    # 파일명에 타임스탬프를 붙이는 이유 — 매 평가 실행마다 별도 파일이 생기므로
    # 이전 평가 결과와 섞이지 않고 실행 이력을 시간순으로 추적할 수 있다.
    log_file = os.path.join(log_dir, f'eval_{time.strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 콘솔 핸들러: 터미널에 실시간 출력하여 진행 상황을 즉시 확인할 수 있게 한다.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 포맷터: 로그 메시지 앞에 시간을 붙여 언제 기록되었는지 알 수 있게 한다.
    # 평가 도중 어느 시점에 이상치가 발생했는지 시간 흐름으로 파악할 수 있다.
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 두 핸들러를 모두 등록 — 파일 기록과 콘솔 출력이 동시에 이루어진다.
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def read_from_file(path):
    """
    테스트 데이터 파일명 리스트가 담긴 텍스트 파일을 읽어 파일명 리스트로 반환한다.

    왜 이 함수가 필요한가?
        split_data.py가 train/val/test 분할 결과를 텍스트 파일로 저장한다.
        평가 시에는 test 데이터에 해당하는 파일명만 필요하므로 이 파일을 읽어
        리스트로 만든다. .png 확장자가 없는 경우 자동으로 붙여주는 이유는,
        분할 시 파일명만 저장하는 경우와 확장자까지 저장하는 경우를 모두
        호환하기 위해서이다.
    """
    data = []

    with open(path, 'r') as file:
        for line in file:
            # 앞뒤 공백/줄바꿈 제거 — 텍스트 파일 편집 시 실수로 들어간
            # 공백이 파일명 오류를 일으키지 않도록 정리한다.
            line = line.strip()
            if line:  # 빈 줄은 건너뛴다 — 파일 끝의 빈 줄 등이 오류를 유발하지 않도록.
                # 파일명에 .png 확장자가 이미 있으면 그대로 쓰고, 없으면 붙여준다.
                # 이렇게 하면 확장자 유무에 상관없이 동일한 방식으로 처리할 수 있다.
                image_file = line if line.endswith('.png') else line + '.png'
                # 파일명만 리스트에 추가한다.
                data.append(image_file)

    return data


def log_gpu_info(logger):
    """
    사용 가능한 GPU의 이름과 메모리 사용량을 로그에 기록한다.

    왜 GPU 정보를 기록하는가?
        평가 결과는 사용한 GPU의 메모리 상태에 영향을 받을 수 있다(예: OOM).
        나중에 결과를 재현하거나 문제를 분석할 때 어떤 GPU 환경에서 실행했는지
        알 수 있도록 환경 정보를 로그에 남긴다.
    """
    if torch.cuda.is_available():
        # 시스템에 여러 GPU가 있을 수 있으므로 모든 GPU에 대해 정보를 기록한다.
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            # 현재 할당된 메모리 — 평가 시작 시점의 베이스라인으로 활용한다.
            logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
            # 캐시된 메모리 — PyTorch가 재사용을 위해 예약해 둔 메모리 양이다.
            logger.info(f"Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")


def save_prediction_npy(pred, point_cloud, img_names, save_dir='./eval_npy', logger=None):
    """
    예측 포인트 클라우드를 .npy 파일로 저장하고, 항목별 바운딩 박스/부피/IoU를 계산해 로그에 기록한다.

    왜 저장과 동시에 지표를 계산하는가?
        배치 단위로 이미 메모리에 올라와 있는 예측/정답 텐서를 이용해 즉시
        지표를 계산하면, 나중에 .npy 파일을 다시 읽어올 필요가 없어 효율적이다.
        또한 항목별 지표를 로그에 남겨두면 어떤 샘플에서 성능이 좋고 나쁜지
        개별적으로 추적할 수 있어 오류 분석에 유리하다.
    """
    # 배치 내 각 샘플에 대해 순회하며 저장 + 지표 계산을 수행한다.
    for i, (pred_points, gt_points) in enumerate(zip(pred, point_cloud)):
        # 예측 포인트 클라우드를 CPU로 옮긴 뒤 numpy 배열로 변환한다.
        # 왜 CPU로 옮기는가? np.save는 numpy 배열만 받으므로, GPU 텐서를
        # 직접 저장할 수 없다. .cpu()로 먼저 옮긴 후 변환한다.
        pred_np = pred_points.cpu().numpy()
        img_name = img_names[i]
        # 확장자를 제거한 파일명 — .png 대신 .npy로 저장하기 위해.
        filename = os.path.splitext(img_name)[0]
        save_path = os.path.join(save_dir, f'{filename}.npy')
        # 예측 포인트 클라우드를 NPY 파일로 저장한다.
        np.save(save_path, pred_np)

        # 바운딩 박스 및 볼륨 계산
        # 예측/정답 각각의 3D 바운딩 박스(최소/최대 좌표)를 구한다.
        # 바운딩 박스 부피는 물류 박스의 실제 크기를 추정하는 데 직접적으로
        # 활용되므로, 예측 부피와 정답 부피를 비교하여 모델의 크기 추정
        # 정확도를 평가한다.
        pred_min, pred_max = calculate_3d_bbox(pred_points)
        gt_min, gt_max = calculate_3d_bbox(gt_points)

        # 바운딩 박스로부터 부피를 계산한다.
        pred_volume = calculate_volume(pred_min, pred_max)
        gt_volume = calculate_volume(gt_min, gt_max)

        # 개별 IOU 계산
        # calculate_3d_miou는 배치 차원이 있는 텐서를 기대하므로, 단일 샘플에
        # 대해 호출할 때는 배치 차원을 추가해 주어야 한다(unsqueeze(0)).
        # 이렇게 하면 [1, N, 3] 형태가 되어 함수 내부의 배치 처리 로직이
        # 정상 동작한다.
        pred_unsqueeze = pred_points.unsqueeze(0)  # [1, N, 3]
        gt_unsqueeze = gt_points.unsqueeze(0)      # [1, N, 3]
        iou = calculate_3d_miou(pred_unsqueeze, gt_unsqueeze)

        # 결과 로깅
        if logger:
            # 확장자 제거 — 파일명에서 .png 부분을 떼어낸다.
            name_without_ext = os.path.splitext(img_name)[0]  # 확장자 제거하고 ("01010110_8808041102026_2_3")
            # 파일명 끝의 "_2_3" 같은 접미사를 제거하여 상품 식별 ID만 남긴다.
            # 이 접미사는 동일 상품의 여러 뷰/인스턴스를 구분하기 위한 것이므로,
            # 로그에서는 상품 단위로 집계하기 위해 제거한다.
            display_name = name_without_ext[:-4]  # _2_3 부분 제거해서 ("01010110_8808041102026")
            logger.info(f"Item : {display_name}, pred_vol: {pred_volume.item():.4f}, gt_vol: {gt_volume.item():.4f}, IOU: {iou.item():.4f}")


def main():
    """
    모델을 로드하고 테스트 데이터로 평가를 수행한다.

    전체 흐름:
      1) 명령행 인자 파싱(경로, 배치 크기 등)
      2) 디바이스(GPU/CPU) 설정 및 로거 초기화
      3) 테스트 데이터 로더 생성
      4) 학습된 모델 로드 및 평가 모드 전환
      5) tqdm으로 진행률을 표시하며 배치 단위 추론
      6) 항목별 .npy 저장 + IoU/부피 계산 및 로깅
      7) 최종 평균 IoU 산출 및 로깅
    """
    # ── 명령행 인자 파싱 ──────────────────────────────────────────
    # 왜 argparse를 쓰는가: 하드코딩된 경로를 매번 코드를 수정하지 않고
    # 실행 시점에 지정할 수 있어, Windows/Linux 어디서든 유연하게 실행 가능
    import argparse as _argparse
    parser = _argparse.ArgumentParser(
        description='A-Point-Set-Generation 평가 스크립트')
    # 이미지 루트 경로 — 전처리된 폴리곤 crop PNG들이 있는 디렉토리.
    parser.add_argument('--image_root', default='datasets/images_poly_bbox_crop', type=str,
                        help='이미지 데이터 루트 경로 (상대/절대 경로 모두 가능)')
    # 포인트 클라우드 루트 경로 — 정답(GT) .npy 파일들이 있는 디렉토리.
    parser.add_argument('--point_cloud_root', default='datasets/labels', type=str,
                        help='포인트 클라우드 데이터 루트 경로 (상대/절대 경로 모두 가능)')
    # 모델 체크포인트 경로 — best 검증 loss로 저장된 모델을 사용하는 것이
    # 일반적이다(과적합된 최종 모델보다 best 모델이 일반화 성능이 좋음).
    parser.add_argument('--model_path', default='best-Baseline_DL_Vis.pt', type=str,
                        help='학습된 모델 체크포인트 경로')
    # 테스트 파일명 리스트 — split_data.py가 생성한 test_data.txt.
    parser.add_argument('--test_list', default='datasets/test_data.txt', type=str,
                        help='테스트 데이터 파일명 리스트 (기본: test_data.txt)')
    # 예측 결과 저장 디렉토리 — 항목별 .npy 파일이 여기에 저장된다.
    parser.add_argument('--save_dir', default='./eval_npy', type=str,
                        help='예측 포인트 클라우드 저장 디렉토리 (기본: ./eval_npy)')
    # 배치 크기 — GPU 메모리에 맞춰 조정. 클수록 처리 속도가 빠르지만
    # 메모리 부족(OOM)이 발생할 수 있다.
    parser.add_argument('--batch_size', default=32, type=int,
                        help='배치 크기 (기본: 32)')
    # DataLoader 워커 수 — 멀티프로세싱으로 데이터 로딩을 병렬화한다.
    # Windows에서는 0으로 설정하는 것이 안정적이다(멀티프로세싱 제약 때문).
    parser.add_argument('--num_workers', default=8, type=int,
                        help='DataLoader 워커 수 (기본: 8, Windows에서는 0 권장)')
    # 생성할 포인트 수 — 모델 학습 시와 동일한 값(11000)을 사용해야 한다.
    # 학습 시의 점 수와 다르면 모델의 마지막 레이어 차원이 맞지 않아 오류 발생.
    parser.add_argument('--num_points', default=11000, type=int,
                        help='포인트 클라우드 점 수 (기본: 11000)')
    args = parser.parse_args()

    # 경로를 절대 경로로 변환 (상대 경로도 처리 가능)
    # 왜 절대 경로로 변환하는가? 상대 경로는 현재 작업 디렉토리에 따라
    # 의미가 달라져 혼동을 일으킬 수 있다. 절대 경로로 변환해 로그에
    # 기록하면 정확히 어떤 경로를 사용했는지 추적할 수 있다.
    image_root = os.path.abspath(args.image_root)
    point_cloud_root = os.path.abspath(args.point_cloud_root)
    model_path = os.path.abspath(args.model_path)
    save_dir = args.save_dir

    # 시작 시간과 실행 명령어 기록
    # 실행 명령어를 로그에 남기는 이유: 나중에 어떤 파라미터 조합으로
    # 평가가 수행되었는지 정확히 재현할 수 있도록 하기 위함이다.
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    command = ' '.join(sys.argv)

    # Device setup
    # CUDA가 사용 가능하면 GPU를, 아니면 CPU를 사용한다.
    # GPU가 있을 때 empty_cache()를 호출하는 이유는 이전 실행에서 남아있던
    # 캐시된 메모리를 정리하여 평가 시작 시 메모리 상태를 깨끗하게 만들기 위함이다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create eval_npy directory if it doesn't exist
    # 예측 결과를 저장할 디렉토리가 없으면 생성한다.
    # 저장 단계에서 디렉토리가 없으면 오류가 발생하므로 미리 만들어 둔다.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 로거 설정
    logger = setup_logger()

    # 실행 정보 로깅
    # 구분선과 함께 주요 설정 값을 기록하여 로그 파일만 봐도 실행 조건을
    # 파악할 수 있도록 한다.
    logger.info('='*50)
    logger.info(f'Evaluation started at: {start_time}')
    logger.info(f'Execution command: {command}')
    logger.info(f'image_root: {image_root}')
    logger.info(f'point_cloud_root: {point_cloud_root}')
    logger.info(f'model_path: {model_path}')
    logger.info('='*50)

    # Configuration
    batch_size = args.batch_size
    num_workers = args.num_workers
    # use_2048: 데이터 로더 내부에서 포인트 클라우드를 2048개로 서브샘플링할지
    # 여부를 결정하는 플래그. True로 설정하면 정답 포인트 클라우드 중 2048개만
    # 사용하여 지표 계산의 기준으로 삼는다(연산량 감소 목적).
    use_2048 = True
    # img_size = 227: SqueezeNet 1.1 백본의 구조상 고정된 입력 크기이다.
    # SqueezeNet의 컨볼루션/풀링 구조가 227×227 입력을 기준으로 설계되어 있어
    # 다른 크기를 사용하면 feature map 크기가 어긋나 모델이 동작하지 않는다.
    img_size = 227
    num_points = args.num_points

    # GPU 정보 로깅
    log_gpu_info(logger)
    logger.info('='*50)

    # Load validation data
    # 테스트 파일명 리스트를 읽어온다. 이 리스트가 곧 평가 대상 샘플 목록이다.
    val_data = read_from_file(args.test_list)
    logger.info(f"Total Items: {len(val_data)}")

    # Transform setup
    # 학습 시와 동일한 전처리를 적용해야 한다 — 그렇지 않으면 모델이
    # 학습 때와 다른 입력 분포를 받아 성능이 왜곡된다.
    # interpolation=2는 양선형(bilinear) 보간을 의미한다.
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=2),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    # 데이터 로더 생성 — shuffle=False로 설정하는 이유:
    # 평가 시에는 순서가 고정되어야 각 배치의 파일명과 예측 결과를 정확히
    # 매칭할 수 있다. 셔플하면 파일명 인덱싱이 어긋나 잘못된 결과가 저장된다.
    val_loader = get_loader(
        image_root, point_cloud_root, val_data, use_2048,
        transform, batch_size, False, num_workers
    )

    # Load model
    # 모델 구조를 먼저 생성한 뒤 체크포인트를 로드한다.
    model = pic2points(num_points=num_points)
    # 다중 GPU 환경에서는 DataParallel로 래핑해야 모델이 여러 GPU에 분산되어
    # 로드될 수 있다. 단일 GPU면 이 블록은 건너뛴다.
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 모델 전체를 파일에서 로드한다. 학습 시 torch.save(model)로 저장했으므로
    # 로드 시에도 모델 클래스 정의(pic2points)가 메모리에 있어야 한다.
    # map_location=device는 체크포인트가 다른 디바이스(GPU↔CPU)에서
    # 저장되었을 때 현재 디바이스로 안전하게 옮기기 위함이다.
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    # eval() 모드로 전환 — Dropout을 비활성화하고 BatchNorm을 추론 모드로
    # 설정하여, 평가 시 결정론적(deterministic) 결과를 얻기 위함이다.
    model.eval()

    logger.info(f"Model loaded from: {model_path}")
    logger.info("\nStarting validation...")

    # 평균 IoU 계산을 위한 누적 변수들
    total_iou = 0
    total_samples = 0

    # torch.no_grad(): 평가 시에는 역전파가 필요 없으므로 기울기 계산을
    # 비활성화하여 메모리 사용량과 연산 시간을 크게 줄인다.
    with torch.no_grad():
        # tqdm으로 진행률 표시 — 전체 배치 수 대비 현재 진행 상황과
        # 예상 남은 시간을 보여주어 장시간 평가의 진행을 가늠할 수 있다.
        for i, (image, point_cloud) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Get batch image filenames
            #img_names = [name[1] for name in val_data[i * batch_size:(i + 1) * batch_size]]
            # 현재 배치에 해당하는 파일명 슬라이스 — 저장 시 파일명으로 사용한다.
            # shuffle=False이므로 val_data 순서와 배치 순서가 일치한다.
            img_names = val_data[i * batch_size:(i + 1) * batch_size]
            # 이미지와 정답 포인트 클라우드를 float 텐서로 변환 후 디바이스로 이동.
            image = image.float().to(device)
            point_cloud = point_cloud.float().to(device)

            # 모델 추론 — 입력 이미지로부터 예측 포인트 클라우드를 생성한다.
            pred = model(image)

            # Save predictions and calculate IOU
            # 예측 결과를 파일로 저장하고 항목별 지표를 로그에 기록한다.
            save_prediction_npy(pred, point_cloud, img_names, save_dir, logger)

            # Calculate batch IOU for progress tracking
            # 배치 단위 IoU를 계산하여 누적한다. .item()으로 스칼라 값을 꺼내고,
            # 배치 내 샘플 수(len(image))를 곱해 합산한 뒤 나중에 샘플 수로
            # 나누어 가중 평균을 구한다(단순 평균과 거의 같지만 마지막 배치가
            # 작을 경우 정확한 평균을 보장한다).
            batch_iou = calculate_3d_miou(pred, point_cloud)
            total_iou += batch_iou.item() * len(image)
            total_samples += len(image)

            # 10배치마다 중간 평균 IoU를 로그에 기록하여 진행 상황을 모니터링한다.
            # 매 배치마다 로그를 남기면 로그가 너무 길어지므로 10배치 간격으로 절충.
            if i % 10 == 0:
                avg_iou = total_iou / total_samples
                logger.info(f"Batch [{i+1}/{len(val_loader)}] Average IOU so far: {avg_iou:.4f}")

    # 최종 로깅
    # 전체 샘플에 대한 평균 IoU를 계산하여 최종 결과로 기록한다.
    final_avg_iou = total_iou / total_samples
    logger.info('='*50)
    logger.info(f"\nEvaluation Completed")
    logger.info(f"Total Items processed: {total_samples}")
    logger.info(f"Final Average IOU: {final_avg_iou:.4f}")
    logger.info('='*50)


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────
# 아래는 이전 버전의 코드 블록들이다. 현재는 사용하지 않지만 참고용으로
# 주석 처리하여 유지한다. 코드 로직 변경 없이 원본 그대로 보존한다.
# ──────────────────────────────────────────────────────────────────────────

# def setup_logger():
#     # logs 디렉토리 생성
#     log_dir = './logs'
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     logger = logging.getLogger('evaluation')
#     logger.setLevel(logging.INFO)

#     # 파일 핸들러
#     log_file = os.path.join(log_dir, f'eval_{time.strftime("%Y%m%d_%H%M%S")}.log')
#     fh = logging.FileHandler(log_file)
#     fh.setLevel(logging.INFO)

#     # 콘솔 핸들러
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)

#     # 포맷터
#     formatter = logging.Formatter('%(asctime)s - %(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)

#     logger.addHandler(fh)
#     logger.addHandler(ch)

#     return logger

# def log_gpu_info(logger):
#     if torch.cuda.is_available():
#         for i in range(torch.cuda.device_count()):
#             logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#             logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
#             logger.info(f"Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")

# def save_prediction_npy(pred, point_cloud, img_names, save_dir='./eval_npy', logger=None):
#     for i, (pred_points, gt_points) in enumerate(zip(pred, point_cloud)):
#         # npy 파일 저장
#         pred_np = pred_points.cpu().numpy()
#         img_name = img_names[i]
#         filename = os.path.splitext(img_name)[0]
#         save_path = os.path.join(save_dir, f'{filename}.npy')
#         np.save(save_path, pred_np)

#         # 바운딩 박스 및 볼륨 계산
#         pred_min, pred_max = calculate_3d_bbox(pred_points)
#         gt_min, gt_max = calculate_3d_bbox(gt_points)

#         pred_volume = calculate_volume(pred_min, pred_max)
#         gt_volume = calculate_volume(gt_min, gt_max)

#         # 개별 IOU 계산
#         pred_unsqueeze = pred_points.unsqueeze(0)  # [1, N, 3]
#         gt_unsqueeze = gt_points.unsqueeze(0)      # [1, N, 3]
#         iou = calculate_3d_miou(pred_unsqueeze, gt_unsqueeze)

#         # 결과 로깅
#         if logger:
#             logger.info(f"Image: {img_name}, pred_vol: {pred_volume.item():.4f}, gt_vol: {gt_volume.item():.4f}, IOU: {iou.item():.4f}")

# def main():
#     # 시작 시간과 실행 명령어 기록
#     start_time = time.strftime("%Y-%m-%d %H:%M:%S")
#     command = ' '.join(sys.argv)

#     # Device setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # Create eval_npy directory if it doesn't exist
#     save_dir = './eval_npy'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 로거 설정
#     logger = setup_logger()

#     # 실행 정보 로깅
#     logger.info('='*50)
#     logger.info(f'Evaluation started at: {start_time}')
#     logger.info(f'Execution command: {command}')
#     logger.info('='*50)

#     # Configuration
#     image_root = "/workspace/DATA/nia_tta/image_crop_poly" #"/workspace/DATA/NIA29_3D/images_poly_bbox_crop"
#     point_cloud_root = "/workspace/DATA/nia_tta/pcd" #"/workspace/DATA/NIA29_3D/labels/"
#     batch_size = 32
#     num_workers = 8
#     use_2048 = True
#     img_size = 227
#     num_points = 11000

#     # GPU 정보 로깅
#     log_gpu_info(logger)
#     logger.info('='*50)

#     # Load validation data
#     val_data = read_from_file('/workspace/DATA/nia_tta_processing_code/tta_test.txt') #test_data.txt
#     logger.info(f"Total validation images: {len(val_data)}")

#     # Transform setup
#     transform = transforms.Compose([
#         transforms.Resize(img_size, interpolation=2),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor()
#     ])

#     val_loader = get_loader(
#         image_root, point_cloud_root, val_data, use_2048,
#         transform, batch_size, False, num_workers
#     )

#     # Load model
#     model = pic2points(num_points=num_points)
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)

#     model_path = '/workspace/MODELS/A-Point-Set-Generation-Network-for-3D-Object-Reconstruction-from-a-Single-Image/best-Baseline_DL_Vis.pt'
#     model = torch.load(model_path)
#     model.to(device)
#     model.eval()

#     logger.info(f"Model loaded from: {model_path}")
#     logger.info("\nStarting validation...")

#     total_iou = 0
#     total_samples = 0

#     with torch.no_grad():
#         for i, (image, point_cloud) in enumerate(tqdm(val_loader, desc="Evaluating")):
#             # Get batch image filenames
#             img_names = [name[1] for name in val_data[i * batch_size:(i + 1) * batch_size]]

#             image = image.float().to(device)
#             point_cloud = point_cloud.float().to(device)

#             pred = model(image)

#             # Save predictions and calculate IOU
#             save_prediction_npy(pred, point_cloud, img_names, save_dir, logger)

#             # Calculate batch IOU for progress tracking
#             batch_iou = calculate_3d_miou(pred, point_cloud)
#             total_iou += batch_iou.item() * len(image)
#             total_samples += len(image)

#             if i % 10 == 0:
#                 avg_iou = total_iou / total_samples
#                 logger.info(f"Batch [{i+1}/{len(val_loader)}] Average IOU so far: {avg_iou:.4f}")

#     # 최종 로깅
#     final_avg_iou = total_iou / total_samples
#     logger.info('='*50)
#     logger.info(f"\nEvaluation Completed")
#     logger.info(f"Total images processed: {total_samples}")
#     logger.info(f"Final Average IOU: {final_avg_iou:.4f}")
#     logger.info('='*50)

# if __name__ == "__main__":
#     main()

###################로그 출력 안하게 하려면 아래 주석을 푸세요###################

"""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_loader import get_loader
from split_data import read_from_file
from chamfer_distance import ChamferDistance
from metrics import calculate_3d_miou
from pic2points_model import pic2points
import numpy as np


def save_prediction_npy(pred, img_names, save_dir='./eval_npy'):

    #Save predictions to .npy files for each image in the batch.

    for i, pred_points in enumerate(pred):
        pred_np = pred_points.cpu().numpy()
        img_name = img_names[i]
        filename = os.path.splitext(img_name)[0]
        save_path = os.path.join(save_dir, f'{filename}.npy')
        np.save(save_path, pred_np)

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create eval_npy directory if it doesn't exist
    save_dir = './eval_npy'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Configuration
    image_root = "/workspace/DATA/NIA29_3D/images_poly_bbox_crop"
    point_cloud_root = "/workspace/DATA/NIA29_3D/labels/"
    batch_size = 32
    num_workers = 8
    use_2048 = True
    img_size = 227
    num_points = 11000

    # Transform setup
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=2),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    # Load validation data
    val_data = read_from_file('val_data.txt')
    print(f"Total validation images: {len(val_data)}")

    val_loader = get_loader(
        image_root, point_cloud_root, val_data, use_2048,
        transform, batch_size, False, num_workers
    )

    # Load model
    model = pic2points(num_points=num_points)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = torch.load('/workspace/MODELS/A-Point-Set-Generation-Network-for-3D-Object-Reconstruction-from-a-Single-Image/best-Baseline_DL_Vis.pt')
    model.to(device)
    model.eval()

    # Initialize Chamfer Distance
    chamferDist = ChamferDistance()

    # Validation
    print("Starting validation...")
    total_val_loss = 0
    total_val_miou = 0

    with torch.no_grad():
        for i, (image, point_cloud) in enumerate(val_loader):
            # Get batch image filenames
            img_names = [name[1] for name in val_data[i * batch_size:(i + 1) * batch_size]]

            image = image.float().to(device)
            point_cloud = point_cloud.float().to(device)

            pred = model(image)

            # Save predictions as .npy files
            save_prediction_npy(pred, img_names, save_dir)

            # Calculate losses
            dist = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist[0]) + torch.mean(dist[1])) / 100.0
            miou = calculate_3d_miou(pred, point_cloud)

            total_val_loss += loss.item()
            total_val_miou += miou.item()

            if i % 10 == 0:
                print(f"Batch [{i+1}/{len(val_loader)}] Loss: {loss.item():.4f} mIoU: {miou.item():.4f}")

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_miou = total_val_miou / len(val_loader)

    print(f"\nValidation Results:")
    print(f"Average Loss: {avg_val_loss:.4f}")
    print(f"Average mIoU: {avg_val_miou:.4f}")

if __name__ == "__main__":
    main()
"""
