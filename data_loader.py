# =============================================================================
# data_loader.py — 학습/검증용 데이터 로더
# -----------------------------------------------------------------------------
# 이 파일은 A-Point-Set-Generation 프로젝트에서 PyTorch 학습/검증 파이프라인이
# 사용할 데이터를 공급하는 역할을 담당한다.
#
# 핵심 구성:
#   - XDataset   : torch.utils.data.Dataset 을 상속한 커스텀 데이터셋 클래스.
#                  단일 상품 이미지(PNG)와 그에 대응하는 3D 포인트 클라우드(NPY)를
#                  한 쌍으로 묶어서 반환한다.
#   - get_loader : 위 Dataset 을 감싸서 DataLoader 를 손쉽게 만들어주는 편의 함수.
#
# 데이터 흐름:
#   train_data.txt / val_data.txt 등의 '파일명 리스트'를 입력받아,
#   os.walk 로 이미지 루트와 포인트 클라우드 루트 하위를 순회하면서
#   각 파일명에 대응하는 실제 파일 경로를 미리 매핑(인덱싱)해 둔다.
#   이렇게 미리 매핑해 두면 __getitem__ 호출 시마다 디렉토리를 다시 탐색하지
#   않아도 되므로 학습 속도가 크게 향상된다.
#
# 참고: 전처리된 datasets/ 디렉토리 구조를 사용한다.
#   - 이미지: datasets/images_poly_bbox_crop/<파일명>.png
#   - 포인트 클라우드: datasets/labels/npy_stride5/<파일명>.npy
# =============================================================================

# --- 임포트 ---
# torch: PyTorch 텐서 연산의 기본 패키지. 모델 입력/출력은 모두 torch.Tensor 로 다룬다.
import torch

# torchvision.transforms: 이미지 전처리(Resize, CenterCrop, ToTensor 등)를
# 파이프라인 형태로 조합할 수 있게 해주는 모듈. 학습 시 입력 이미지를
# 모델이 기대하는 크기/형식으로 변환하기 위해 사용한다.
import torchvision.transforms as transforms

# torch.utils.data: Dataset 과 DataLoader 같은 데이터 로딩 유틸리티를 제공.
# 커스텀 Dataset 을 만들고 이를 배치 단위로 묶어 모델에 공급하는 핵심 모듈이다.
import torch.utils.data as data

# os: 파일 경로 조합(os.path.join) 및 디렉토리 순회(os.walk)를 위해 사용.
# 전처리된 데이터가 하위 디렉토리에 흩어져 있으므로 경로 처리가 필수적이다.
import os

# PIL.Image: 이미지를 PIL 형식으로 불러오기 위해 사용.
# torchvision transform 이 PIL 이미지를 입력으로 받기 때문에 PIL 로 로드한다.
from PIL import Image

# numpy: 포인트 클라우드(.npy) 파일을 numpy 배열로 로드하기 위해 사용.
# 포인트 클라우드는 (N, 3) 형태의 좌표 배열이며, numpy 가 .npy 포맷을 지원한다.
import numpy as np

# glob: 파일 패턴 매칭용. 현재 활성 버전에서는 os.walk 를 주로 사용하지만,
# 파일 검색 유틸리티로 함께 임포트되어 있다.
import glob

# random: 데이터 셔플 등 무작위 처리를 위해 임포트. DataLoader 의 shuffle
# 옵션과 별개로 필요한 경우를 대비해 둔다.
import random


#####검증 데이터 evaluation, testing(visualize) 시 활성화#####
# -----------------------------------------------------------------------------
# 아래 주석 처리된 XDataset / get_loader 는 '이전 버전' 구현이다.
# 검증(evaluation)이나 시각화(testing) 단계에서는 이전 버전을 활성화하여
# 사용할 수 있도록 의도적으로 주석으로 보존해 둔 것이다.
#
# 이전 버전과 현재 버전의 차이:
#   - 이전 버전: __getitem__ 에서 매번 os.path.join 으로 경로를 조합하고
#                파일 존재 여부를 즉시 확인한다. 초기화 시 파일 경로를
#                미리 매핑하지 않는다.
#   - 현재 버전: __init__ 에서 os.walk 로 모든 파일 경로를 미리 매핑하여
#                딕셔너리에 저장하고, __getitem__ 에서는 이를 즉시 조회한다.
#                이로 인해 학습 루프에서 디스크 I/O 탐색 비용이 줄어든다.
# -----------------------------------------------------------------------------
# class XDataset(data.Dataset):
#     def __init__(self, image_root, point_cloud_root, id_pairs, transform=None, use_2048=True):
#         # 이미지/포인트 클라우드 루트 경로와 전처리 transform, 포인트 수 제한 옵션을 저장.
#         self.image_root = image_root
#         self.point_cloud_root = point_cloud_root
#         self.transform = transform
#         self.use_2048 = use_2048
#         # ToTensor 만 수행하는 기본 정규화 파이프라인. ImageNet 표준 정규화는 사용하지 않는다.
#         self.normalize = transforms.Compose([transforms.ToTensor()])
#
#         # id_pairs 를 집합(set) 과 리스트 양쪽으로 보관하여 중복 제거 및 순서 보존을 동시에 지원.
#         self.id_pairs_set = set(id_pairs)
#         self.id_pairs = id_pairs
#         self.len = len(id_pairs)
#
#     def __getitem__(self, index):
#         # id_pairs 의 각 원소는 (image_type, specific_type) 형태의 튜플로,
#         # 포인트 클라우드 식별자와 이미지 파일명을 함께 담고 있다.
#         image_type, specific_type = self.id_pairs[index]
#
#         # 이미지 경로 처리
#         image_path = os.path.join(self.image_root, specific_type)
#
#         # 이미지가 존재하는지 확인
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image file not found: {image_path}")
#
#         # 이미지 로드
#         try:
#             # RGB 3채널로 변환하여 로드. 흑백 이미지나 투명도 채널이 있어도 3채널로 통일.
#             image = Image.open(image_path).convert('RGB')
#         except Exception as e:
#             raise Exception(f"Error loading image {image_path}: {str(e)}")
#
#         if self.transform is not None:
#             image = self.transform(image)
#
#         # 포인트 클라우드 경로 처리
#         # 포인트 클라우드 파일명은 image_type 에 .npy 확장자를 붙여 구성한다.
#         point_cloud_path = os.path.join(self.point_cloud_root, image_type + '.npy')
#
#         # 포인트 클라우드가 존재하는지 확인
#         if not os.path.exists(point_cloud_path):
#             raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")
#
#         # 포인트 클라우드 로드
#         try:
#             point_cloud = np.load(point_cloud_path)
#         except Exception as e:
#             raise Exception(f"Error loading point cloud {point_cloud_path}: {str(e)}")
#
#         return image, point_cloud
#
#     def __len__(self):
#         return self.len
#
# def get_loader(image_root, point_cloud_root, id_pairs, use_2048, transform, batch_size, shuffle, num_workers):
#     # XDataset 인스턴스를 생성한 뒤 DataLoader 로 감싸서 배치 단위 데이터를 공급.
#     dataset = XDataset(image_root, point_cloud_root, id_pairs, transform=transform, use_2048=use_2048)
#
#     data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         # drop_last=False: 마지막 배치가 batch_size 보다 작아도 버리지 않고 그대로 사용.
#         # 검증/테스트 시에는 모든 샘플을 평가해야 하므로 마지막 배치도 유지한다.
#         drop_last=False
#     )
#
#     return data_loader

#####학습 시 활성화 #####


class XDataset(data.Dataset):
    """
    학습/검증용 커스텀 PyTorch Dataset.

    단일 상품 이미지(PNG)와 그에 대응하는 3D 포인트 클라우드(NPY)를 한 쌍으로
    묶어서 반환한다. 초기화 시점에 os.walk 로 전체 파일 경로를 미리 매핑해 두어,
    학습 루프 중 __getitem__ 이 호출될 때마다 디스크를 다시 탐색하는 비용을
    제거한다. 이는 대규모 데이터셋에서 학습 속도에 직접적인 영향을 주는 최적화다.

    입력:
        - image_root        : 이미지가 저장된 최상위 디렉토리 경로.
        - point_cloud_root  : 포인트 클라우드(.npy)가 저장된 최상위 디렉토리 경로.
        - id_pairs          : 학습/검증에 사용할 파일명 리스트 (train_data.txt 등).
        - transform         : 이미지 전처리 파이프라인(Resize, CenterCrop, ToTensor 등).
        - use_2048          : 포인트 수를 2048로 제한할지 여부 (현재 활성 로직에서는
                              직접 사용되지 않지만, 이전 버전 호환성 및 향후 실험을
                              위해 인터페이스에 남겨둔다).

    반환 (__getitem__):
        - image       : 전처리가 적용된 이미지 텐서 (transform 이 None 이면 PIL 이미지).
        - point_cloud : (N, 3) numpy 배열 — N 개 점의 3D 좌표.
    """

    def __init__(self, image_root, point_cloud_root, id_pairs, transform=None, use_2048=True):
        # 루트 경로들을 인스턴스 변수로 저장. __getitem__ 에서 파일을 로드할 때 사용한다.
        self.image_root = image_root
        self.point_cloud_root = point_cloud_root

        # 이미지 전처리 파이프라인을 저장. None 이면 전처리 없이 PIL 이미지를 그대로 반환.
        self.transform = transform

        # 포인트 수 제한 옵션. 현재 활성 로직에서는 직접 사용되지 않지만,
        # 이전 버전과의 인터페이스 호환성 및 향후 포인트 수 실험을 위해 유지한다.
        self.use_2048 = use_2048

        # ToTensor 만 수행하는 기본 변환 파이프라인을 별도로 구성.
        # ImageNet 표준 정규화(mean/std)를 사용하지 않는 것은 원본 코드의 설계 선택이며,
        # SqueezeNet 입력이 [0,1] 범위의 텐서를 기대하기 때문이다.
        self.normalize = transforms.Compose([transforms.ToTensor()])

        # 학습/검증에 사용할 파일명 리스트를 저장.
        # 이 리스트의 순서가 __getitem__ 의 index 와 1:1 로 대응된다.
        self.id_pairs = id_pairs

        # 데이터셋 크기를 미리 계산하여 __len__ 에서 즉시 반환할 수 있도록 한다.
        self.len = len(id_pairs)

        # 이미지와 포인트 클라우드 파일의 '전체 경로'를 미리 매핑해 두는 딕셔너리.
        # 키는 파일명(예: "01010101_8801039920121.png"), 값은 실제 전체 경로.
        # __getitem__ 에서 O(1) 조회가 가능해지므로 학습 속도가 향상된다.
        self.image_paths = {}
        self.point_cloud_paths = {}

        # 인덱싱 시작을 알리는 로그. 대규모 데이터셋에서는 시간이 걸릴 수 있어 진행 상황을 알린다.
        print("Indexing files...")
        for filename in self.id_pairs:
            # 확장자를 제거한 '기본 이름'을 추출.
            # 포인트 클라우드 파일명은 이미지 파일명에서 확장자만 .npy 로 바꾼 형태이므로
            # 기본 이름이 두 파일을 연결하는 공통 키 역할을 한다.
            basename = os.path.splitext(filename)[0]

            # --- 이미지 파일 찾기 ---
            # 이미지 루트 하위의 모든 하위 디렉토리를 재귀적으로 순회한다.
            # 전처리 결과가 images_poly_bbox_crop/ 바로 아래가 아닌 하위 폴더에
            # 흩어져 있을 수 있으므로 os.walk 로 전체 트리를 탐색한다.
            image_found = False
            for root, _, files in os.walk(self.image_root):
                if filename in files:
                    # 파일을 찾으면 전체 경로를 딕셔너리에 저장하고 탐색을 중단.
                    self.image_paths[filename] = os.path.join(root, filename)
                    image_found = True
                    break

            # --- 포인트 클라우드 파일 찾기 (.npy) ---
            # 포인트 클라우드 파일명은 '기본 이름 + .npy' 형태다.
            pc_found = False
            pc_filename = basename + '.npy'
            # 포인트 클라우드 루트 하위도 재귀적으로 순회한다.
            # labels/npy_stride5/ 와 같이 하위 디렉토리에 저장되어 있기 때문이다.
            for root, _, files in os.walk(self.point_cloud_root):
                if pc_filename in files:
                    self.point_cloud_paths[filename] = os.path.join(root, pc_filename)
                    pc_found = True
                    break

            # 파일이 누락된 경우 경고를 출력한다.
            # 학습에 사용할 수 없는 샘플이므로 사용자가 데이터를 점검할 수 있도록 알린다.
            if not image_found:
                print(f"Warning: Image not found for {filename}")
            if not pc_found:
                print(f"Warning: Point cloud not found for {basename}.npy")

        # 이미지와 포인트 클라우드가 '모두' 존재하는 유효한 쌍의 개수를 집계.
        # 실제 학습에 사용 가능한 샘플 수를 사용자에게 알려주기 위함이다.
        valid_pairs = sum(1 for f in self.id_pairs
                         if f in self.image_paths and f in self.point_cloud_paths)
        print(f"Found {valid_pairs} image-pointcloud pairs")

    def __getitem__(self, index):
        """인덱스에 해당하는 이미지-포인트클라우드 쌍을 반환한다."""
        # id_pairs 리스트에서 index 번째 파일명을 가져온다.
        filename = self.id_pairs[index]

        # 미리 매핑해 둔 경로 딕셔너리에서 해당 파일의 전체 경로를 조회한다.
        # __init__ 에서 인덱싱을 수행했기 때문에 여기서는 디스크 탐색 없이 O(1) 로 가져온다.
        image_path = self.image_paths.get(filename)
        if image_path is None:
            # 경로가 없다는 것은 인덱싱 단계에서 파일을 찾지 못했음을 의미.
            raise FileNotFoundError(f"Image file not found: {filename}")

        # PIL 로 이미지를 열고 RGB 3채널로 변환.
        # convert('RGB') 를 하는 이유: 흑백 이미지나 RGBA(투명도 포함) 이미지가 섞여 있어도
        # 모델 입력 채널 수를 3으로 통일하기 위해서다.
        image = Image.open(image_path).convert('RGB')

        # 전처리 파이프라인이 지정된 경우 적용 (Resize, CenterCrop, ToTensor 등).
        # transform 이 None 이면 PIL 이미지가 그대로 반환되므로 호출하는 쪽에서 주의해야 한다.
        if self.transform is not None:
            image = self.transform(image)

        # 포인트 클라우드 경로도 동일하게 미리 매핑된 딕셔너리에서 조회.
        point_cloud_path = self.point_cloud_paths.get(filename)
        if point_cloud_path is None:
            raise FileNotFoundError(f"Point cloud file not found for: {filename}")

        # .npy 파일을 numpy 배열로 로드. (N, 3) 형태의 3D 좌표 배열이다.
        # numpy 배열은 텐서로 변환되기 전의 원시 형태이며, 호출하는 쪽(train_ajit.py 등)에서
        # 필요에 따라 torch.from_numpy() 로 변환하여 모델 입력으로 사용한다.
        point_cloud = np.load(point_cloud_path)

        # (이미지, 포인트 클라우드) 쌍을 반환.
        # DataLoader 가 이 튜플들을 배치 단위로 묶어 모델에 공급한다.
        return image, point_cloud

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환한다."""
        # __init__ 에서 미리 계산해 둔 길이를 반환.
        # DataLoader 가 전체 배치 수를 계산할 때 이 값을 사용한다.
        return self.len


def get_loader(image_root, point_cloud_root, id_pairs, use_2048, transform, batch_size, shuffle, num_workers):
    """
    XDataset 을 감싸는 DataLoader 를 생성하여 반환하는 편의 함수.

    학습/검증 스크립트(baseline_main.py 등)에서 직접 DataLoader 를 조립하지 않고
    이 함수 하나로 데이터 로딩 파이프라인을 구성할 수 있도록 제공된다.

    매개변수:
        - image_root        : 이미지 최상위 경로.
        - point_cloud_root  : 포인트 클라우드 최상위 경로.
        - id_pairs          : 파일명 리스트 (train_data.txt / val_data.txt 의 내용).
        - use_2048          : 포인트 수 제한 옵션 (XDataset 의 인터페이스 호환용).
        - transform         : 이미지 전처리 파이프라인.
        - batch_size        : 한 배치당 샘플 수. 메모리와 학습 안정성에 직접적 영향.
        - shuffle           : 에폭마다 데이터 순서를 섞을지 여부. 학습 시 True 로 설정하여
                              모델이 데이터 순서에 의존하지 않도록 일반화 성능을 높인다.
        - num_workers       : 데이터 로딩에 사용할 병렬 프로세스 수. I/O 병목을 줄여준다.
                              (Windows 환경에서는 0 으로 설정하는 것이 안정적이다.)

    반환:
        - data_loader : 학습/검증 루프에서 직접 순회할 수 있는 DataLoader 인스턴스.
    """
    # XDataset 인스턴스를 생성. 이 시점에서 os.walk 로 모든 파일 경로가 인덱싱된다.
    dataset = XDataset(image_root, point_cloud_root, id_pairs, transform=transform, use_2048=use_2048)

    # DataLoader 로 Dataset 을 감싸서 배치 단위로 데이터를 공급하도록 구성.
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # drop_last=False: 마지막 배치가 batch_size 보다 작아도 버리지 않고 그대로 사용.
        # 학습 시에도 모든 샘플을 활용하며, 검증/테스트 시에는 특히 누락 없이 평가해야 한다.
        drop_last=False
    )

    return data_loader
