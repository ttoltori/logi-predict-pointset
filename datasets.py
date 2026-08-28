# =============================================================================
# PartDataset — ShapeNet 파트 세그멘테이션 데이터셋을 위한 PyTorch Dataset 클래스
# -----------------------------------------------------------------------------
# 본 파일은 A-Point-Set-Generation 원본 프로젝트의 ShapeNet PartDataset 클래스다.
# 현재 NIA29 물류 3D 데이터 파이프라인에서는 사용되지 않지만,
# train_ajit.py 가 import 하므로 호환성 유지를 위해 그대로 보존한다.
#
# [데이터 구조]
#   synsetoffset2category.txt  : 카테고리 이름(Chair, Table, ...) ↔ synset ID(03001627, ...)
#   <synset_id>/points/       : 각 3D 모델의 포인트 클라우드 (.pts)
#   <synset_id>/points_label/ : 각 점의 파트 라벨 (.seg)
#   <synset_id>/expert_verified/seg_img/ : 렌더링된 2D 이미지 (.png)
#
# [주의] cv2 (OpenCV) 를 사용하므로 opencv-python 패키지가 필요하다.
# =============================================================================

# Python 2 와 3 의 print 함수 호환성을 위해 사용 — 구형 코드베이스와의 호환 유지 목적
from __future__ import print_function

# PyTorch 의 Dataset 기반 클래스 — 커스텀 데이터셋은 반드시 data.Dataset 을 상속해야
# DataLoader 가 __getitem__ / __len__ 을 자동으로 호출할 수 있다.
import torch.utils.data as data

# PIL 은 이미지를 파이썬 객체로 다루기 위한 라이브러리 — 원본 코드에서 import 되어 있으나
# 실제로는 cv2(OpenCV) 가 이미지 로딩에 사용되므로 여기서는 참조용으로 남아 있음.
from PIL import Image

# 파일 경로 조작 및 디렉토리 탐색을 위한 표준 라이브러리들
import os
import os.path
import errno

# PyTorch 텐서 변환 및 모델 입력 준비를 위해 import
import torch

# JSON 파일 읽기/쓰기를 위해 import — 원본에서 메타데이터 처리에 사용될 수 있음
import json

# 텍스트 인코딩 처리용 — ShapeNet 의 일부 메타데이터가 UTF-8 으로 저장되어 있을 수 있어 포함
import codecs

# 포인트 클라우드 배열 연산 및 이미지 통계(평균/표준편차) 계산에 사용
import numpy as np

# 표준 출력 제어용 — 원본 코드의 디버그 출력 호환성 유지
import sys

# 이미지 전처리 파이프라인(ToTensor, Normalize 등) 구성에 사용
import torchvision.transforms as transforms

# 커맨드라인 인자 파싱용 — 원본 스크립트에서 독립 실행 시 옵션 지정에 사용될 수 있음
import argparse

# json 이 위에서 이미 import 되었으나 원본 코드에 중복 import 가 존재 — 로직 변경 없이 그대로 유지
import json

# OpenCV — 이미지를 numpy 배열로 로딩하고 리사이즈/색상반전 처리하기 위해 사용.
# PIL 대신 cv2 를 사용하는 이유는 bitwise_not 등 배열 기반 연산이 직관적이고 빠르기 때문.
import cv2


class PartDataset(data.Dataset):
    """
    ShapeNet 파트 세그멘테이션 데이터셋을 위한 PyTorch Dataset 클래스.

    이 클래스는 단일 2D 이미지로부터 3D 포인트 클라우드를 생성하는
    'A Point Set Generation Network' 원본 논문의 데이터 로딩을 담당한다.

    역할:
      - ShapeNet 카테고리(Chair, Table, Airplane 등)별로 포인트 클라우드(.pts),
        파트 라벨(.seg), 렌더링 이미지(.png) 파일 경로를 수집한다.
      - train/val 을 9:1 비율로 분할한다.
      - __getitem__ 호출 시 이미지를 227×227 로 리사이즈하고 색상을 반전한 뒤
        정규화하여 텐서로 변환하고, 포인트 클라우드를 npoints 만큼 랜덤 샘플링하여 반환한다.

    주의:
      - 현재 NIA29 데이터에는 사용되지 않지만 train_ajit.py 가 import 하므로 유지된다.
      - cv2 (OpenCV) 를 사용하므로 opencv-python 패키지 설치가 필요하다.
    """

    def __init__(self, root, npoints=2500, pic2point=True, class_choice=None, train=True):
        """
        데이터셋을 초기화하여 모든 파일 경로를 사전 수집한다.
        (Dataset 은 생성 시점에 전체 파일 리스트를 메모리에 올려두는 것이 일반적이다.
         매 __getitem__ 호출마다 디렉토리를 탐색하면 I/O 오버헤드가 크기 때문이다.)
        """
        # 샘플링할 포인트 수 — 포인트 클라우드의 원본 점 개수가 다를 수 있으므로
        # 고정된 개수로 통일하여 미니배치 텐서 크기를 일관되게 유지한다.
        self.npoints = npoints

        # ShapeNet 데이터셋의 루트 디렉토리 경로
        self.root = root

        # synsetoffset2category.txt: 카테고리 이름과 ShapeNet synset ID 를 매핑하는 파일.
        # 예: "Chair 03001627" → cat["Chair"] = "03001627"
        # 이 파일이 있어야 어떤 하위 폴더가 어떤 물체 카테고리인지 알 수 있다.
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')

        # 카테고리 이름 → synset ID 매핑 딕셔너리
        # 예: {"Chair": "03001627", "Table": "04379243", ...}
        self.cat = {}  # a searching dictionary

        # pic2point 모드 여부 — True 이면 (이미지, 포인트클라우드) 쌍을 반환하고,
        # False 이면 (포인트클라우드, 세그멘테이션라벨, 클래스ID) 를 반환한다.
        # 이미지→3D 재구성(task 1)과 포인트클라우드 파트 세그멘테이션(task 2)을
        # 하나의 클래스에서 전환할 수 있도록 설계되었다.
        self.pic2point = pic2point

        # 카테고리 매핑 파일을 읽어 딕셔너리를 채운다.
        # 각 줄은 "카테고리이름 synsetID" 형식이다.
        with open(self.catfile, 'r') as f:
            for line in f:
                # 줄 끝의 공백/개행을 제거하고 공백을 기준으로 분리
                ls = line.strip().split()
                # 예: self.cat["Chair"] = "03001627"
                self.cat[ls[0]] = ls[1]  # cat[Chair] = 03001627

        # class_choice 가 지정된 경우, 해당 카테고리만 필터링한다.
        # 예: class_choice=["Chair"] 이면 Chair 카테고리 데이터만 사용.
        # 특정 물체 카테고리만으로 학습/실험하고 싶을 때 사용한다.
        if not class_choice is  None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # meta 딕셔너리: 각 카테고리별로 (포인트경로, 라벨경로, 이미지경로) 튜플의 리스트를 저장.
        # 예: meta["Chair"] = [("/.../points/xxx.pts", "/.../points_label/xxx.seg", "/.../seg_img/xxx.png"), ...]
        self.meta = {}  # meta[03001627] [pts, seg]
        for item in self.cat:

            self.meta[item] = []

            # 각 카테고리 폴더 내의 하위 디렉토리 경로를 구성한다.
            # points/       : 3D 포인트 클라우드 파일 (.pts)
            # points_label/ : 각 점이 어떤 파트(예: 의자 다리, 등받이)에 속하는지 라벨 (.seg)
            # expert_verified/seg_img/ : 2D 렌더링 이미지 (.png)
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_pic = os.path.join(self.root, self.cat[item], 'expert_verified/seg_img')

            # 이미지 파일 목록을 정렬하여 가져온다.
            # sorted() 를 사용하는 이유: 파일 탐색 순서가 OS 에 따라 달라질 수 있으므로
            # 정렬하여 재현 가능한(train/val 분할이 항상 동일한) 결과를 보장하기 위함.
            fns = sorted(os.listdir(dir_pic))

            # train/val 을 9:1 비율로 분할한다.
            # 정렬된 파일 리스트의 앞 90% 를 train, 뒤 10% 를 val 로 사용한다.
            # 무작위 셔플이 아닌 정렬 기준 분할이므로, 데이터가 알파벳 순으로 편향될 수 있으나
            # 원본 코드의 설계를 그대로 유지한다.
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            # 각 이미지 파일에 대응하는 포인트 클라우드(.pts)와 라벨(.seg) 경로를 매핑한다.
            # 이미지 파일명에서 확장자를 제거한 token 이 .pts/.seg 파일명과 동일하다는 가정 하에 동작.
            for fn in fns:
                # 파일명에서 확장자(.png)를 제거하여 token 추출
                token = (os.path.splitext(os.path.basename(fn))[0])
                # (포인트경로, 라벨경로, 이미지경로) 튜플을 meta 리스트에 추가
                self.meta[item].append((os.path.join(dir_point, token + '.pts'),
                                        os.path.join(dir_seg, token + '.seg'),
                                        os.path.join(dir_pic, token + '.png')))

        # datapath: 모든 카테고리의 데이터를 하나의 평탄한 리스트로 병합한다.
        # 각 원소는 (카테고리이름, 포인트경로, 라벨경로, 이미지경로) 튜플이다.
        # 이 리스트의 인덱스가 __getitem__ 의 index 인자가 된다.
        self.datapath = []  # car,  xxx_v0/02691156/xxxx.pts, xxx_v0/02691156/xxxx.seg, xxx_v0/02691156/xxxx.png
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        # 카테고리 이름을 정수 인덱스로 매핑 — 모델이 클래스를 원핫이 아닌
        # 정수 ID 로 받을 수 있도록 한다. 예: {"Chair": 0, "Table": 1, ...}
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # 파트 세그멘테이션 클래스 수 — pic2point=False (세그멘테이션 모드) 일 때만 계산.
        # 모든 데이터를 확인하면 느리므로 50 개 간격으로 샘플링하여 최대 클래스 수를 추정한다.
        self.num_seg_classes = 0
        if not self.pic2point:
            # 원본 코드: len(self.datapath)/50 — Python 2 에서는 정수 나눗셈, Python 3 에서는 float.
            # 50 개마다 하나씩 샘플링하여 라벨 파일의 고유값 개수를 확인한다.
            for i in range(len(self.datapath) / 50):
                # 라벨 파일(.seg)을 로드하여 고유한 파트 ID 개수를 센다.
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        """
        주어진 인덱스의 데이터(이미지 텐서, 포인트 클라우드)를 반환한다.
        DataLoader 가 배치를 구성할 때 이 메서드를 반복적으로 호출한다.
        """
        # index 에 해당하는 데이터 경로 튜플을 가져온다.
        fn = self.datapath[index]

        # 카테고리 이름을 정수 클래스 ID 로 변환한다.
        cls = self.classes[self.datapath[index][0]]

        # 포인트 클라우드 파일(.pts)을 numpy 배열로 로드한다.
        # .pts 파일은 텍스트 형식으로 각 줄이 "x y z" 좌표이며, np.loadtxt 가 이를 파싱한다.
        # float32 로 변환하는 이유: PyTorch 모델의 가중치와 동일한 정밀도를 사용하여
        # 불필요한 타입 캐스팅 오버헤드를 피하기 위함.
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        #seg = np.loadtxt(fn[2]).astype(np.int64)

        # OpenCV 로 이미지를 numpy 배열(BGR 순서)로 로드한다.
        im = cv2.imread(fn[3])

        # 이미지가 정상적으로 로드된 경우에만 전처리를 수행한다.
        # cv2.imread 는 파일이 없거나 손상된 경우 None 을 반환하므로 이를 확인한다.
        if (type(im) != type(None)):

            # 227×227 로 리사이즈 — SqueezeNet 등의 입력 크기에 맞추기 위함.
            # 227 는 원본 AlexNet/SqueezeNet 의 표준 입력 해상도이다.
            im = cv2.resize(im, (227, 227))

            # bitwise_not 으로 색상을 반전한다.
            # ShapeNet 렌더링 이미지는 배경이 흰색(255), 객체가 어두운 색이므로
            # 반전하면 배경이 검은색(0), 객체가 밝은 색이 되어
            # 신경망이 객체 영역의 특징을 더 잘 추출할 수 있다.
            im = cv2.bitwise_not(im)

            # 이미지의 평균과 표준편차를 계산하여 정규화 파라미터로 사용한다.
            # 255 로 나누는 이유: cv2 이미지는 0~255 범위의 uint8 이므로
            # 0~1 범위로 스케일링한 뒤 정규화하기 위함.
            # ImageNet 의 전역 mean/std 가 아닌 개별 이미지의 통계를 사용하는 것은
            # 원본 코드의 설계 선택이다.
            mean = im.mean(axis=(0, 1, 2)) / 255
            std = im.std(axis=(0, 1, 2)) / 255

            # torchvision transforms 로 전처리 파이프라인을 구성한다.
            # ToTensor: H×W×C numpy 배열을 C×H×W 텐서로 변환하고 0~1 로 스케일링.
            # Normalize: (x - mean) / std 로 각 채널을 정규화하여 학습 안정성을 높인다.
            # 동일한 mean/std 값을 3 채널(R,G,B) 모두에 적용한다.
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((mean, mean, mean), (std, std, std))])

            # 구성된 transform 을 이미지에 적용하여 최종 텐서를 얻는다.
            im = transform(im)

        #print(point_set.shape, im.shape)

        # resample
            # 포인트 클라우드에서 npoints 개의 점을 무작위로 복원 추출(random with replacement)한다.
            # 복원 추출을 사용하는 이유: 원본 점 개수가 npoints 보다 적은 경우에도
            # 중복을 허용하여 npoints 개를 채울 수 있기 때문이다.
            # 매 epoch 마다 다른 점이 샘플링되므로 데이터 증강(augmentation) 효과도 있다.
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)  # random choice n id for selection
            point_set = point_set[choice, :]
        #seg = seg[choice]

            # numpy 배열을 PyTorch 텐서로 변환한다.
            point_set = torch.from_numpy(point_set)


        #seg = torch.from_numpy(seg)
            # 클래스 ID 도 텐서로 변환 — 모델 출력과 타입을 맞추기 위해 int64 로 캐스팅.
            cls = torch.from_numpy(np.array([cls]).astype(np.int64))

            # pic2point 모드: (이미지, 포인트클라우드) 반환 — 단일 이미지→3D 재구성 태스크
            # 비 pic2point 모드: (포인트클라우드, 세그멘테이션라벨, 클래스ID) 반환 — 파트 세그멘테이션 태스크
            if self.pic2point:
                return im, point_set
            else:
                return point_set, seg, cls

    def __len__(self):
        """
        데이터셋의 전체 샘플 개수를 반환한다.
        DataLoader 가 전체 배치 수를 계산하기 위해 호출한다.
        """
        return len(self.datapath)
