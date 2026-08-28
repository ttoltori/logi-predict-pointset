# =============================================================================
# Volume_estimation.py
# -----------------------------------------------------------------------------
# 목적: 학습된 pic2points 모델로 단일 상품 이미지에서 3D 포인트 클라우드를
#       예측하고, 예측된 점들의 x/y/z 좌표 범위로 박스 치수(가로/세로/높이)를
#       추정한 뒤 JSON 라벨에 기록된 실제 치수와 비교하여 부피를 계산한다.
#
# 왜 이 스크립트가 필요한가:
#   - 물류 현장에서는 상품의 부피를 알아야 적재·보관·배송 비용을 산정할 수 있다.
#   - 실물을 직접 측정하지 않고 상품 이미지 한 장만으로 부피를 추정하면
#     작업 시간과 인건비를 크게 줄일 수 있다.
#   - 이 스크립트는 모델 예측 결과가 실제 부피와 얼마나 차이가 나는지를
#     검증하여, 모델이 실제 서비스에 투입될 수 있는지 평가하는 역할을 한다.
# =============================================================================

# --- 표준 라이브러리 ---
# os: 파일 경로 조작(절대 경로 변환, 디렉토리 생성 등)을 위해 사용.
#     Windows/Linux 모두에서 경로 구분자 문제 없이 동작시키기 위함이다.
import os
# argparse: 실행 시점에 데이터 경로·모델 경로 등을 명령행 인자로 전달받기 위해 사용.
#           코드를 수정하지 않고 다양한 환경(로컬/서버)에서 재사용할 수 있다.
import argparse
# json: 상품의 실제 치수(width/length/height)가 저장된 JSON 라벨 파일을 읽기 위해 사용.
import json
# math: 숫자를 10 단위로 내림(floor)하기 위해 사용. 치수를 규격화된 값으로 맞추기 위함이다.
import math
# glob: 이미지 디렉토리 하위의 모든 PNG 파일을 재귀적으로 찾기 위해 사용.
from glob import glob

# --- 수치/데이터 처리 ---
# numpy: 포인트 클라우드(N×3 배열)의 좌표별 최소/최대값 계산 등 벡터 연산을 위해 사용.
import numpy as np
# pandas: 부피 추정 결과를 표 형태로 정리하여 CSV 파일로 저장하기 위해 사용.
import pandas as pd
# tqdm: 추론 루프의 진행 상황을 시각적으로 확인하기 위해 사용.
#        테스트 데이터가 많을 때 남은 시간을 파악할 수 있다.
from tqdm import tqdm

# --- PyTorch ---
# torch: 텐서 연산 및 GPU/CPU 디바이스 선택을 위해 사용.
import torch
# torch.nn: 다중 GPU 환경에서 모델을 병렬화하는 DataParallel을 사용하기 위해 import.
import torch.nn as nn
# Variable: (레거시) 텐서를 자동미분 가능한 변수로 감싸는 용도.
#           최신 PyTorch에서는 잘 쓰이지 않지만 원본 코드 호환성을 위해 유지.
from torch.autograd import Variable
# transforms: 이미지를 모델 입력 크기(227×227)로 resize/crop하고 텐서로 변환하기 위해 사용.
import torchvision.transforms as transforms

# --- 프로젝트 내부 모듈 ---
# get_loader: 학습/검증용 데이터 로더를 생성하는 함수.
#             이미지와 포인트 클라우드를 쌍으로 묶어 배치 단위로 제공한다.
from data_loader import get_loader
# read_from_file: test_data.txt 등의 텍스트 파일에서 테스트 파일명 리스트를 읽어온다.
from split_data import read_from_file
# ChamferDistance: 예측 점과 정답 점 사이의 거리를 계산하는 손실 함수.
#                  본 스크립트에서는 직접 사용하지 않지만 평가 파이프라인 호환성을 위해 import.
from chamfer_distance import ChamferDistance
# calculate_3d_miou: 3D IoU(교집합/합집합 비율)를 계산하는 평가 지표 함수.
#                    본 스크립트에서는 직접 사용하지 않지만 평가 파이프라인 호환성을 위해 import.
from metrics import calculate_3d_miou
# pic2points: 단일 이미지 → 3D 포인트 클라우드를 생성하는 모델 클래스.
#             SqueezeNet 백본 + 포인트 생성 헤드로 구성된다.
from pic2points_model import pic2points

# open3d는 3D 포인트 클라우드 시각화/처리 라이브러리이다.
# 본 스크립트에서는 실제로 사용하지 않지만, 향후 시각화 기능을 추가할 수 있도록
# import를 시도하고 실패해도 프로그램이 중단되지 않도록 try/except로 감싼다.
# 왜 try/except로 처리하는가: open3d는 Windows에서 설치가 까다로워
# 의존성 문제로 인해 핵심 로직이 실행되지 않는 것을 방지하기 위함이다.
try:
    import open3d as o3d  # 현재 코드에서 미사용, 호환성을 위해 import 시도
except ImportError:
    o3d = None  # open3d가 없어도 실행 가능 (현재 코드에서 미사용)


def modify_value(number):
    """
    숫자를 10 단위로 내림하여 규격화된 치수 값으로 변환한다.

    왜 10 단위로 내림하는가:
        - 포인트 클라우드의 좌표는 연속적인 실수값이므로 예측 결과에 작은 노이즈가 포함된다.
        - 치수를 10mm 단위로 규격화하면 노이즈가 평활화되어 부피 계산이 안정적이 된다.
        - 물류 현장에서도 박스 치수를 10mm 단위로 관리하는 경우가 많아
          실제 업무 규격과 맞추기 위한 목적도 있다.
    """
    # number * 10 / 10 은 소수점 첫째 자리에서 내림하는 효과를 낸다.
    # 예: 123.7 → 1230 / 10 = 123 → 123 * 10 = 1230 (10 단위로 내림)
    # 예: 128.9 → 1280 / 10 = 128 → 128 * 10 = 1280 (10 단위로 내림)
    return math.floor(number * 10 / 10) * 10


def find_json_data(image_path, json_root):
    """
    이미지 경로에서 파일명을 추출하고 해당하는 JSON 파일을 찾아서 치수 정보를 반환한다.

    왜 JSON 라벨이 필요한가:
        - 모델이 예측한 포인트 클라우드만으로는 '실제 상품이 얼마나 큰지'를 알 수 없다.
          모델 출력은 상대적인 3D 형태일 뿐 절대적인 물리적 단위(mm)가 아니기 때문이다.
        - JSON 라벨에는 사람이 직접 측정한 상품의 실제 치수(mm)가 기록되어 있어
          예측 결과를 실제 치수와 비교하여 스케일 보정 비율을 구하는 기준이 된다.

    왜 치수에 10을 곱하는가:
        - JSON 라벨의 단위가 cm로 저장되어 있어, 포인트 클라우드 좌표(mm)와
          단위를 통일하기 위해 10을 곱해 cm → mm 로 변환한다.
    """
    # 이미지 파일명에서 확장자를 제거하고 .json 확장자를 붙여 JSON 파일 경로를 생성한다.
    # 예: "01010101_8801039920121.png" → "01010101_8801039920121.json"
    json_path = os.path.join(json_root, os.path.basename(image_path).split('.')[0] + '.json')

    # JSON 파일이 존재하는 경우에만 치수를 읽어온다.
    # 파일이 없으면 None을 반환하여 호출 측에서 예외가 발생하지 않도록 한다.
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            # COCO-like 형식: annotations 배열의 첫 번째 요소의 attributes에 치수가 저장됨.
            # (attributes 변수는 참조용으로 읽었으나 실제 치수는 아래에서 직접 접근한다.)
            attributes = json_data['annotations'][0]['attributes']

            # 각 치수에 10.0을 곱해 cm → mm 단위로 변환한다.
            # width  = 가로, length = 세로(깊이), height = 높이
            width  = json_data['annotations'][0]['attributes']['width']  * 10.0
            length = json_data['annotations'][0]['attributes']['length'] * 10.0
            height = json_data['annotations'][0]['attributes']['height'] * 10.0
            return width, length, height


def main():
    """
    모델을 로드하고 테스트 데이터로 포인트 클라우드를 예측한 뒤,
    예측된 포인트 클라우드의 x/y/z 범위로 박스 치수를 계산하고
    실제 JSON 치수와 비교하여 부피를 추정하는 메인 함수.
    """

    # ── 명령행 인자 파싱 ──────────────────────────────────────────
    # 왜 argparse를 쓰는가: 하드코딩된 경로를 매번 코드를 수정하지 않고
    # 실행 시점에 지정할 수 있어, Windows/Linux 어디서든 유연하게 실행 가능
    parser = argparse.ArgumentParser(
        description='A-Point-Set-Generation 부피 추정 스크립트')

    # --image_root: 상품 이미지(PNG)가 저장된 최상위 디렉토리.
    #               전처리(prepare_data_3d.py) 결과인 images_poly_bbox_crop 가 기본값.
    parser.add_argument('--image_root', default='datasets/images_poly_bbox_crop', type=str,
                        help='이미지 데이터 루트 경로 (상대/절대 경로 모두 가능)')

    # --point_cloud_root: 정답 포인트 클라우드(NPY)가 저장된 디렉토리.
    #                     본 스크립트에서는 데이터 로더 생성 시 경로 일관성을 위해 전달.
    parser.add_argument('--point_cloud_root', default='datasets/labels', type=str,
                        help='포인트 클라우드 데이터 루트 경로 (상대/절대 경로 모두 가능)')

    # --json_root: 실제 치수가 기록된 JSON 라벨 디렉토리.
    #              스케일 보정 기준값(실제 치수)을 얻기 위해 반드시 필요하다.
    parser.add_argument('--json_root', default='datasets/json_labels', type=str,
                        help='치수 JSON 라벨 루트 경로 (상대/절대 경로 모두 가능)')

    # --model_path: 학습이 완료된 모델 체크포인트(.pt) 경로.
    #               best- 접두사가 붙은 파일은 검증 loss가 가장 낮은 시점의 가중치이다.
    parser.add_argument('--model_path', default='best-Baseline_DL_Vis.pt', type=str,
                        help='학습된 모델 체크포인트 경로')

    # --test_list: 테스트에 사용할 파일명 리스트가 담긴 텍스트 파일.
    #              split_data.py 가 train/val/test 8:1:1 로 분할한 결과 중 test 분할.
    parser.add_argument('--test_list', default='datasets/test_data.txt', type=str,
                        help='테스트 데이터 파일명 리스트 (기본: test_data.txt)')

    # --num_workers: DataLoader 가 데이터를 병렬로 로드할 때 사용하는 프로세스 수.
    #                 Windows에서는 멀티프로세싱이 불안정해 0을 권장한다.
    parser.add_argument('--num_workers', default=8, type=int,
                        help='DataLoader 워커 수 (기본: 8, Windows에서는 0 권장)')

    # --num_points: 모델이 출력할 포인트 클라우드의 점 개수.
    #               학습 시 사용한 값(11000)과 반드시 일치해야 한다.
    parser.add_argument('--num_points', default=11000, type=int,
                        help='포인트 클라우드 점 수 (기본: 11000)')

    # --csv_path: 부피 추정 결과를 저장할 CSV 파일 경로.
    #             엑셀에서 한글이 깨지지 않도록 저장 시 utf-8-sig 인코딩을 사용한다.
    parser.add_argument('--csv_path', default='box_volumes.csv', type=str,
                        help='부피 추정 결과 CSV 저장 경로 (기본: box_volumes.csv)')

    args = parser.parse_args()

    # ── 경로를 절대 경로로 변환 ────────────────────────────────────
    # 왜 절대 경로로 변환하는가: 상대 경로는 현재 작업 디렉토리(cwd)에 의존하여
    # 어디서 스크립트를 실행하느냐에 따라 다르게 해석될 수 있다.
    # 절대 경로로 변환하면 실행 위치에 관계없이 항상 동일한 경로를 가리키게 된다.
    image_root       = os.path.abspath(args.image_root)
    point_cloud_root = os.path.abspath(args.point_cloud_root)
    json_root        = os.path.abspath(args.json_root)
    model_path       = os.path.abspath(args.model_path)

    # ── 디바이스 설정 ─────────────────────────────────────────────
    # 왜 cuda 가 available 한지 확인하는가: GPU 가 있으면 대규모 텐서 연산을
    # CPU 보다 수십~수백 배 빠르게 수행할 수 있기 때문이다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GPU 를 사용하는 경우 캐시된 미사용 메모리를 해제한다.
    # 왜 empty_cache를 호출하는가: 이전 실행에서 남은 GPU 메모리 단편화를 방지하여
    # 추론 중 OOM(Out Of Memory) 오류가 발생할 확률을 낮추기 위함이다.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 결과 저장 디렉토리 생성 ───────────────────────────────────
    # eval_npy 디렉토리는 (평가 스크립트와의 일관성을 위해) 예측 결과를 저장하는 표준 위치이다.
    # 디렉토리가 없으면 생성하여 FileNotFoundError 를 방지한다.
    save_dir = './eval_npy'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ── 설정값 출력 ───────────────────────────────────────────────
    # 왜 출력하는가: 실행 전 경로가 올바른지 사용자가 확인할 수 있도록 하여
    # 잘못된 경로로 인한 무의미한 추론을 사전에 방지하기 위함이다.
    print(f'image_root = {image_root}')
    print(f'point_cloud_root = {point_cloud_root}')
    print(f'json_root = {json_root}')
    print(f'model_path = {model_path}')

    # ── 추론 하이퍼파라미터 설정 ───────────────────────────────────
    # batch_size = 1: 부피 추정은 이미지별로 개별 결과를 저장해야 하므로 배치 크기를 1로 고정한다.
    #                 배치가 커지면 결과와 이미지 경로의 순서 대응이 깨질 수 있기 때문이다.
    batch_size = 1
    num_workers = args.num_workers

    # use_2048: 데이터 로더가 2048개 점을 사용할지 여부.
    #           True 로 설정하면 정답 포인트 클라우드를 2048 점으로 서브샘플링한다.
    #           본 스크립트에서는 정답을 직접 사용하지 않으므로 영향이 적지만
    #           데이터 로더의 기대 입력 형식을 맞추기 위해 True 로 유지한다.
    use_2048 = True

    # img_size = 227: SqueezeNet 1.1 백본의 입력 크기는 227×227 로 고정되어 있다.
    #                 이 값은 모델 구조상 변경할 수 없는 고정값이다.
    img_size = 227

    # num_points: 모델이 출력할 3D 점의 개수. 학습 시 값과 반드시 일치해야 한다.
    num_points = args.num_points

    # ── 이미지 전처리 파이프라인 ───────────────────────────────────
    # 왜 이 순서인가:
    #   1. Resize: 원본 이미지를 227×227 로 확대/축소한다. SqueezeNet 입력 크기에 맞추기 위함.
    #      interpolation=2 는 양선형(bilinear) 보간으로, 품질과 속도의 균형이 좋다.
    #   2. CenterCrop: 중앙을 기준으로 227×227 로 자른다. Resize 후 비율이 맞지 않을 수 있어 보정.
    #   3. ToTensor: PIL 이미지(0~255) → 텐서(0~1) 변환. 픽셀 값을 [0,1] 로 정규화한다.
    #      (본 프로젝트는 ImageNet 표준 정규화를 사용하지 않는 설계 선택을 따른다.)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=2),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    # ── 테스트 데이터 리스트 로드 ──────────────────────────────────
    # test_data.txt 에서 테스트용 파일명(확장자 제외) 리스트를 읽어온다.
    val_data = read_from_file(args.test_list)
    print(f"Total validation images: {len(val_data)}")

    # ── 데이터 로더 생성 ───────────────────────────────────────────
    # get_loader: 이미지와 포인트 클라우드를 쌍으로 묶어 배치 단위로 제공하는 DataLoader 생성.
    # 왜 shuffle=False 인가: 추론 결과를 원본 이미지 경로와 1:1 로 매칭해야 하므로
    # 데이터 순서가 섞이면 안 된다. (여섯 번째 인자 False 가 shuffle=False 를 의미)
    val_loader = get_loader(
        image_root, point_cloud_root, val_data, use_2048,
        transform, batch_size, False, num_workers
    )

    # ── 테스트 이미지 경로 리스트 구성 ─────────────────────────────
    # 왜 glob 으로 다시 파일을 찾는가: 데이터 로더는 이미지 텐서만 반환하고
    # 파일 경로를 주지 않기 때문에, JSON 라벨을 찾으려면 별도로 경로 리스트가 필요하다.
    # val_data 에 포함된 파일명과 일치하는 PNG 파일만 골라낸다.
    target_image_path_list = []
    for path in sorted(glob(image_root + '/**/*.png')):
        if os.path.basename(path) in val_data:
            target_image_path_list.append(path)

        # print(json_root, json_dirs, json_files)

    # ── 모델 로드 ─────────────────────────────────────────────────
    # pic2points: 단일 이미지 → 3D 포인트 클라우드 생성 모델을 생성한다.
    # num_points 는 학습 시 값과 동일해야 출력 텐서 shape 이 일치한다.
    model = pic2points(num_points=num_points)

    # 다중 GPU 환경에서는 DataParallel 로 모델을 감싸면 데이터를 분산 처리할 수 있다.
    # 왜 device_count > 1 인지 확인하는가: GPU 가 1개뿐인데 DataParallel 을 쓰면
    # 오버헤드만 증가하고 속도 이득이 없기 때문이다.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 왜 map_location을 쓰는가: CUDA가 없는 환경에서도 CPU로 모델 로드 가능
    # 학습 시 GPU 로 저장된 체크포인트를 CPU 환경에서 로드할 때
    # 텐서가 CUDA 에 매핑되어 있어 에러가 발생하는 것을 방지한다.
    model = torch.load(model_path, map_location=device, weights_only=False)
    # 모델을 지정한 디바이스(GPU 또는 CPU)로 이동시킨다.
    model.to(device)
    # eval 모드로 전환: 드롭아웃/배치정규화를 추론 모드로 고정한다.
    # 왜 eval() 이 필요한가: 학습 모드에서는 Dropout 이 활성화되고
    # BatchNorm 이 현재 배치 통계를 사용하지만, 추론 시에는 이를 끄고
    # 학습 시 누적된 통계를 사용해야 일관된 결과가 나오기 때문이다.
    model.eval()

    # 결과를 저장할 리스트. 각 이미지별 부피 추정 결과를 딕셔너리로 담는다.
    data = []

    # ── 추론 루프 ─────────────────────────────────────────────────
    # 왜 torch.no_grad() 를 쓰는가: 추론 시에는 역전파가 필요 없으므로
    # 그래디언트 계산을 비활성화해 메모리 사용량을 대폭 줄이고 속도를 높인다.
    with torch.no_grad():
        # tqdm 으로 진행 상황을 표시하며 데이터 로더에서 배치를 하나씩 꺼낸다.
        # batch_size=1 이므로 매 반복마다 이미지 1장과 정답 포인트 클라우드 1개가 반환된다.
        for i, (image, _) in enumerate(tqdm(val_loader)):
            # 정답 포인트 클라우드(_)는 부피 추정에 사용하지 않으므로 무시한다.
            # 왜 무시하는가: 본 스크립트는 JSON 라벨의 실제 치수를 기준으로 삼기 때문이다.

            # 데이터 로더의 i 번째 결과에 대응하는 원본 이미지 경로를 가져온다.
            # 이 경로에서 파일명을 추출해 JSON 라벨을 찾는다.
            image_path = target_image_path_list[i]

            # 해당 이미지에 대한 JSON 데이터 찾기
            # r_box_*: 실제(real) 박스 치수. 스케일 보정의 기준이 된다.
            r_box_width, r_box_length, r_box_height = find_json_data(image_path, json_root)

            # ── 포인트 클라우드 예측 ───────────────────────────────
            # 이미지를 float 텐서로 변환 후 지정한 디바이스로 이동시킨다.
            # 왜 float() 인가: 모델 가중치가 float32 이므로 입력도 동일 타입이어야 한다.
            image = image.float().to(device)

            # 모델에 이미지를 입력해 포인트 클라우드를 예측한다.
            # 출력 shape: (batch=1, num_points, 3) → cpu().numpy()[0] 로 (num_points, 3) 배열 추출.
            # 왜 cpu() 로 옮기는가: numpy 변환은 CPU 텐서만 지원하기 때문이다.
            pred = model(image).cpu().numpy()[0]

            # ── 예측 포인트 클라우드에서 박스 치수 계산 ───────────────
            # 포인트 클라우드의 각 축(x, y, z) 범위(최소~최대)를 박스 치수로 간주한다.
            # 왜 abs(np.min(...)) + np.max(...) 인가:
            #   - 포인트 클라우드는 원점(0,0,0)을 중심으로 퍼져 있을 수 있어
            #     최소값이 음수, 최대값이 양수가 되는 경우가 많다.
            #   - 따라서 음수 방향 폭(abs(min))과 양수 방향 폭(max)을 더해
            #     전체 폭(치수)을 구한다.
            #   - modify_value 로 10 단위 규격화하여 노이즈를 평활화한다.
            p_width  = modify_value(abs(np.min(pred[:, 0]))) + modify_value(np.max(pred[:, 0]))
            p_length = modify_value(abs(np.min(pred[:, 1]))) + modify_value(np.max(pred[:, 1]))
            p_height = modify_value(abs(np.min(pred[:, 2]))) + modify_value(np.max(pred[:, 2]))

            # ── 스케일 보정 ───────────────────────────────────────
            # 왜 평균(mean)을 사용하는가:
            #   - 모델이 출력하는 포인트 클라우드는 상대적 형태는 잘 잡지만
            #     절대적 크기(스케일)는 보장하지 않는다.
            #   - 따라서 예측 치수와 실제 치수의 비율을 구해 예측을 실제 크기로 맞춘다.
            #   - 한 축만 비교하면 해당 축의 예측 오차가 스케일 전체로 퍼지므로
            #     세 축의 평균을 사용해 보정 비율을 안정화한다.
            #
            # mean1: 실제 치수 세 축의 평균 (기준)
            # mean2: 예측 치수 세 축의 평균 (보정 대상)
            mean1 = sum([r_box_width, r_box_length, r_box_height]) / 3.0
            mean2 = sum([p_width, p_length, p_height]) / 3.0

            # 버그 수정: max1/max2 → mean1/mean2 (정의되지 않은 변수 참조 오류)
            # scale = 실제 평균 / 예측 평균 → 예측 치수에 곱하면 실제 크기에 가까워진다.
            scale = mean1 / mean2

            # ── 부피 계산 ─────────────────────────────────────────
            # 실제 부피: JSON 치수의 곱 (mm³)
            actual_volume = int(r_box_width * r_box_length * r_box_height)

            # 예측 부피: 스케일 보정된 예측 치수의 곱 (mm³)
            # scale 을 각 축 치수에 곱한 뒤 세 축을 곱한다.
            predicted_volume = int(scale * p_width * scale * p_length * scale * p_height)

            # ── 결과 출력 ─────────────────────────────────────────
            # 왜 출력하는가: 루프 중간에 진행 상황과 추정 품질을 즉시 확인하여
            # 이상치(부피 차이가 큰 샘플)를 조기에 발견하기 위함이다.
            print(f'Image: {os.path.basename(image_path)}',
                  f'실제 박스 부피: {actual_volume:,}mm³',
                  f'추정 박스 부피: {predicted_volume:,}mm³')

    # ── 결과를 CSV 파일로 저장 ─────────────────────────────────────
    # 왜 CSV 로 저장하는가: 부피 추정 결과를 표 형태로 보관하여
    # 엑셀 등 스프레드시트 프로그램에서 쉽게 분석·공유할 수 있기 때문이다.
    csv_file_path = args.csv_path

    # data 리스트를 DataFrame 으로 변환한다.
    df = pd.DataFrame(data)

    # 왜 utf-8-sig 인코딩인가: 엑셀에서 CSV 를 열 때 한글이 깨지는 것을 방지하기 위해
    # BOM(Byte Order Mark) 이 포함된 utf-8-sig 인코딩을 사용한다.
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {csv_file_path}")


# 스크립트가 직접 실행될 때만 main() 을 호출한다.
# 왜 이 패턴을 쓰는가: 다른 모듈에서 import 할 때 main() 이 자동 실행되는 것을 방지하여
# 이 파일이 라이브러리로도 안전하게 사용될 수 있도록 하기 위함이다.
if __name__ == "__main__":
    main()
