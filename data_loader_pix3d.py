# ============================================================================
# A-Point-Set-Generation 프로젝트 — Pix3D 테스트용 데이터 로더
# ----------------------------------------------------------------------------
# 이 파일은 학습이 끝난 모델을 Pix3D(또는 NIA29) 테스트 데이터로 평가할 때
# 사용하는 데이터 로더입니다. 학습용 data_loader.py 와 달리 파일명 리스트가
# 아닌 "객체 ID 리스트"를 입력받아 이미지-포인트클라우드 쌍을 구성하는 것이
# 핵심 차이점입니다. baseline_main.py 에서 Pix3D 테스트 시 호출됩니다.
# ============================================================================

# JSON 파일을 읽어 들이기 위해 사용합니다 — 원본 ShapeNet 버전에서는
# shapenet_test.json 에 객체별 메타데이터가 JSON 형태로 저장되어 있습니다.
import json

# 여러 키에 대해 리스트를 쉽게 묶어주는 defaultdict 입니다 — 본 파일에서는
# 실제로 사용하지 않지만 원본 코드에서 임포트한 상태로 보존합니다.
from collections import defaultdict

# PyTorch 의 Dataset 베이스 클래스 — 커스텀 데이터셋을 만들 때 상속받아
# __len__ 과 __getitem__ 을 구현하면 DataLoader 가 자동으로 배치 단위로
# 데이터를 묶어 모델에 공급할 수 있습니다.
from torch.utils.data import Dataset
# DataLoader 는 Dataset 을 감싸서 배치 크기만큼 데이터를 나누고, 셔플하고,
# 멀티프로세싱으로 병렬 로딩을 수행하는 역할을 합니다. 본 파일에서는
# DataLoader 를 직접 사용하지 않지만, 호출하는 쪽(baseline_main.py)에서
# 이 TestDataset 을 DataLoader 로 감싸 사용하므로 임포트를 유지합니다.
from torch.utils.data import DataLoader

# torchvision 의 transforms 모듈 — 이미지를 모델 입력 크기에 맞게
# Resize/CenterCrop/ToTensor 하는 전처리 파이프라인을 구성할 때 사용합니다.
import torchvision.transforms as transforms

# numpy — 포인트 클라우드 .npy 파일을 배열로 불러오고 수치 연산을 할 때
# 사용합니다. 포인트 클라우드는 (N, 3) 형태의 numpy 배열입니다.
import numpy as np

# os — 파일 시스템 경로를 다루고, os.walk 로 하위 디렉토리까지 순회하며
# 이미지/포인트클라우드 파일을 검색하기 위해 사용합니다.
import os

# imageio — 원본 코드에서 이미지 입출력용으로 임포트되었으나 현재 활성
# 버전에서는 PIL.Image 가 사용됩니다. 임포트는 보존합니다.
import imageio

# torch — 텐서 연산을 위한 PyTorch 코어 라이브러리입니다. 본 파일에서는
# 직접 텐서를 생성하지 않지만, transforms.ToTensor() 가 반환하는 텐서
# 타입을 위해 임포트되어 있습니다.
import torch

# PIL.Image — 이미지 파일을 열 때 사용합니다. torchvision transforms 가
# PIL 이미지를 입력으로 받기 때문에 imageio 대신 PIL 을 사용합니다.
from PIL import Image

# Path for the dataset:
# For eg. path = "/datasets/cs253-wi20-public/Pix3d/"
# At this address you shall find a folder img with images and a folder model with the pointclouds
# Inside these folders you should have the corresponding images and models for different objects

# ============================================================================
# [참고용 — 원본 ShapeNet 버전 (현재 비활성, 주석 처리됨)]
# ----------------------------------------------------------------------------
# 아래 블록은 원본 ShapeNet 데이터셋을 위한 TestDataset 입니다. NIA29
# 데이터로 마이그레이션하면서 두 번째 블록(활성 버전)으로 대체되었습니다.
# 코드 로직 보존을 위해 주석 상태로 유지합니다.
# ============================================================================
"""
class TestDataset():
    def __init__(self, img_path=None, pc_path=None, objects = [2691156, 2933112, 2958343, 3001627, 3636649, 4256520, 4379243, 4530566]):
        a = json.load(open('/workspace/MODELS/A-Point-Set-Generation-Network-for-3D-Object-Reconstruction-from-a-Single-Image/shapenet_test.json'))
        
        img_size = 227
        self.transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                transforms.CenterCrop(img_size),transforms.ToTensor()])
        
        self.Data = []
        for i in range(len(a)):
            if(a[i]['category'] in objects):
                x = a[i]['model']
                x = x.rsplit('/', 1)[0]
                try:
                    for o in os.listdir(pc_path + str(x)):
                        self.Data.append([pc_path + str(x)+'/'+ o, img_path+a[i]['img'].split('/',1)[1]])
                except:
                    #print("File not found")
                    pass
"""

# ============================================================================
# [활성 버전 — NIA29 물류 3D 데이터용 TestDataset]
# ----------------------------------------------------------------------------
# 객체 ID(예: 'B120110053') 리스트를 받아, 이미지 폴더와 포인트 클라우드
# 폴더를 각각 순회하며 매칭되는 쌍을 찾아 self.Data 에 저장합니다.
# 학습/검증용 Xdataset(data_loader.py)과 달리 파일명 리스트가 아닌
# "객체 ID 리스트"를 기준으로 동작하는 것이 특징입니다.
# ============================================================================
class TestDataset():
    """
    Pix3D / NIA29 테스트용 데이터셋 클래스.

    객체 ID 리스트(예: ['B120110053', 'B150130100', ...])를 입력받아,
    지정된 이미지 경로와 포인트 클라우드 경로에서 각 객체에 대응하는
    PNG 이미지와 NPY 포인트 클라우드 파일을 검색하여 매핑합니다.

    특징:
      - 초기화 시점에 os.walk 로 전체 폴더를 한 번 순회하여 파일 경로를
        미리 self.Data 리스트에 구성해 둡니다. 이는 매 __getitem__ 호출
        마다 파일을 다시 검색하는 오버헤드를 없애기 위함입니다.
      - 이미지는 227x227 크기로 Resize/CenterCrop 됩니다. 227 는
        SqueezeNet 1.1 백본의 고정 입력 크기이므로 임의로 변경할 수 없습니다.
      - 포인트 클라우드는 .npy 파일에서 numpy 배열로 그대로 로드합니다.
        (학습 시와 동일하게 별도의 텐서 변환을 하지 않습니다.)
    """

    def __init__(self, img_path=None, pc_path=None, objects = ['B120110053', 'B150130100', 'B160130070', 'B170130120', 'B180180100', 'B200100100', 'B200130140', 'B200180120', 'B200200200', 'B220180120', 'B230200090', 'B240160150', 'B250130065', 'B250154104', 'B250160120', 'B250180060', 'B250210160', 'B250230150', 'B250250260', 'B260160140', 'B260180135', 'B260260260', 'B265215120', 'B270200120', 'B270240100', 'B280170105', 'B280180190', 'B280200060', 'B290260200', 'B300140100', 'B300185115', 'B300200080', 'B300200110', 'B300200120', 'B300230060', 'B300240130', 'B300250080', 'B300250150', 'B300260260', 'B300280100', 'B300300250', 'B305205150', 'B310200195', 'B310220210', 'B310235070', 'B315220300', 'B315250210', 'B320240080', 'B320240130', 'B330180160', 'B330220160', 'B330220220', 'B335170110', 'B340250210', 'B340250250', 'B340260100', 'B340260300', 'B340270250', 'B340285100', 'B350230150', 'B350250140', 'B350270230', 'B350350150', 'B350350200', 'B355305300', 'B360240230', 'B360250200', 'B360250300', 'B360300100', 'B360300150', 'B370250250', 'B380210135', 'B380250260', 'B380270080', 'B380280140', 'B390280140', 'B395320210', 'B400200150', 'B400210320', 'B400250200', 'B400300100', 'B400300130', 'B400310210', 'B400310310', 'B400315215', 'B400340100', 'B400350120', 'B400405300', 'B410310280', 'B420435190', 'B450300100', 'B450320250', 'B450320300', 'B460350350', 'B480385315', 'B480390320', 'B500300300', 'B500320220', 'B500350250', 'B520450400', 'B520480400', 'B540320170', 'B540390340', 'B550400400', 'B600400250', 'B600500300', 'B600500450', 'B700500300', 'B700500500']):  # objects list 생략
        # 객체 ID 는 박스 크기 코드입니다(예: B120110053 = 가로120×세로110×높이53).
        # 이 ID 가 파일명(PNG, NPY)과 일치하므로 검색 키로 사용합니다.

        # 이미지/포인트클라우드 루트 경로를 인스턴스 변수로 보관합니다.
        # os.walk 시 반복적으로 참조해야 하기 때문에 미리 저장해 둡니다.
        self.img_path = img_path
        self.pc_path = pc_path

        # SqueezeNet 1.1 백본은 입력 이미지가 227x227 이어야만 동작합니다.
        # 따라서 img_size 를 227 로 고정합니다 — 이 값은 모델 구조상
        # 변경이 불가능한 하드 제약조건입니다.
        img_size = 227

        # 이미지 전처리 파이프라인을 구성합니다.
        #  1) Resize(227, interpolation=2): 짧은 변을 227 로 맞춥니다.
        #     interpolation=2 는 PIL 의 BILINEAR 보간법으로, 속도와 품질의
        #     균형이 좋아 테스트 단계에서 주로 사용됩니다.
        #  2) CenterCrop(227): 중앙을 기준으로 227x227 정사각형으로 자릅니다.
        #     상품 이미지는 보통 bbox crop 되어 있어 중앙에 객체가 있으므로
        #     CenterCrop 이 정보 손실을 최소화합니다.
        #  3) ToTensor(): PIL 이미지(HWC, 0~255) → 텐서(CHW, 0.0~1.0) 변환.
        #     픽셀 값을 255 로 나누어 [0,1] 범위로 정규화합니다.
        # 참고: ImageNet 표준 mean/std 정규화는 적용하지 않습니다 — 이는
        # 원본 프로젝트의 설계 선택이며 학습 시에도 동일하게 적용됩니다.
        self.transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                transforms.CenterCrop(img_size),transforms.ToTensor()])

        # 최종적으로 (npy_path, img_path) 쌍을 저장할 리스트입니다.
        # __getitem__ 에서 이 리스트를 인덱스로 접근하여 데이터를 반환합니다.
        self.Data = []

        # 이미지와 포인트 클라우드 파일의 전체 경로를 미리 매핑
        # 초기화 단계에서 한 번 전체 폴더를 순회하여 매핑해 두면,
        # 이후 DataLoader 가 __getitem__ 을 호출할 때 파일 검색 비용이
        # 들지 않아 학습/평가 속도가 크게 향상됩니다.
        print("Indexing test files...")

        # 객체 ID 하나당 하나의 이미지-포인트클라우드 쌍을 찾습니다.
        for filename in objects:
            # 이미지 파일 찾기
            # 플래그 변수 — 이미지를 찾았는지 여부를 추적하여, 찾지 못한
            # 경우 사용자에게 경고를 출력하기 위해 사용합니다.
            image_found = False

            # os.walk 로 이미지 루트 하위의 모든 디렉토리를 재귀 순회합니다.
            # 하위 디렉토리 구조가 일정하지 않은 NIA29 데이터 특성상,
            # 특정 폴더만 지정하지 않고 전체를 순회하는 방식이 안전합니다.
            for root, _, files in os.walk(self.img_path):
                # 객체 ID + ".png" 형태의 파일명을 찾습니다.
                png_filename = filename + ".png"
                if png_filename in files:
                    # 매칭된 이미지의 전체 경로를 구성합니다.
                    img_path = os.path.join(root, png_filename)
                    image_found = True

                    # 대응하는 포인트 클라우드 파일 찾기
                    # 이미지를 찾았더라도 포인트 클라우드가 없으면 평가가
                    # 불가능하므로, 포인트 클라우드도 동일하게 검색합니다.
                    pc_found = False
                    for pc_root, _, pc_files in os.walk(self.pc_path):
                        # 객체 ID + ".npy" 형태의 포인트 클라우드 파일을 찾습니다.
                        npy_filename = filename + ".npy"
                        if npy_filename in pc_files:
                            # 매칭된 포인트 클라우드의 전체 경로를 구성합니다.
                            npy_path = os.path.join(pc_root, npy_filename)
                            # (npy_path, img_path) 순서로 저장합니다.
                            # __getitem__ 에서 self.Data[idx][0] 은 npy,
                            # self.Data[idx][1] 은 img 가 되도록 맞춘 것입니다.
                            self.Data.append([npy_path, img_path])
                            pc_found = True
                            # 포인트 클라우드는 객체당 하나만 있으면 되므로
                            # 찾는 즉시 내부 루프를 빠져나옵니다.
                            break

                    # 포인트 클라우드를 찾지 못한 경우 — 데이터 누락이 평가
                    # 결과에 조용히 반영되는 것을 방지하기 위해 경고를 출력합니다.
                    if not pc_found:
                        print(f"Warning: Point cloud not found for {filename}")
                    # 이미지도 찾았으므로 외부 루프(이미지 검색)도 종료합니다.
                    break

            # 이미지 자체를 찾지 못한 경우 — 마찬가지로 경고를 출력하여
            # 데이터 준비 단계의 문제를 사용자가 인지할 수 있게 합니다.
            if not image_found:
                print(f"Warning: Image not found for {filename}")

        # 최종적으로 구성된 유효한 테스트 쌍의 개수를 출력합니다.
        # 이 값이 0 이면 평가를 진행할 수 없으므로 즉시 확인이 필요합니다.
        print(f"Found {len(self.Data)} valid test pairs")

    def __len__(self):
        # 데이터셋의 전체 샘플 수를 반환합니다 — DataLoader 가 전체
        # 배치 개수를 계산할 때 이 값을 참조합니다.
        return len(self.Data)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 테스트 샘플(이미지, 포인트클라우드)을 반환합니다.
        # DataLoader 가 배치를 구성할 때 인덱스별로 이 메서드를 호출합니다.

        # 포인트 클라우드를 numpy 배열로 로드합니다.
        # .npy 파일은 (N, 3) 형태이며 N=11000 개 점의 (x,y,z) 좌표입니다.
        # 별도의 텐서 변환을 하지 않고 numpy 그대로 반환하는 이유는,
        # 평가 단계(metrics.py 등)에서 numpy 기반 연산을 수행하기
        # 때문입니다. 모델 입력으로는 이미지만 텐서가 필요합니다.
        pointcloud = np.load(self.Data[idx][0])

        # 이미지를 PIL 로 열고 RGB 3채널로 변환합니다.
        # convert('RGB') 를 하는 이유는 일부 PNG 가 RGBA(투명도 채널)나
        # 그레이스케일일 수 있는데, SqueezeNet 입력은 3채널로 고정되어
        # 있기 때문에 강제로 3채널로 맞춰주기 위함입니다.
        img = Image.open(self.Data[idx][1]).convert('RGB')

        # 전처리 파이프라인이 설정되어 있으면 이미지에 적용합니다.
        # Resize → CenterCrop → ToTensor 순서로 변환되어 (3, 227, 227)
        # 텐서가 됩니다. 이 텐서가 모델의 입력으로 들어갑니다.
        if self.transform is not None:
            img = self.transform(img)

        # [이미지 텐서, 포인트 클라우드 numpy 배열] 을 반환합니다.
        # 평가 코드에서는 이미지로 예측 포인트 클라우드를 생성하고,
        # 정답 포인트 클라우드와 비교하여 IoU/Chamfer Distance 를 계산합니다.
        return [img, pointcloud]
