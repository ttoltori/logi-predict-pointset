# =============================================================================
# 본 파일은 A-Point-Set-Generation 프로젝트의 모델 정의 파일입니다.
#
# 전체 흐름:
#   1) 단일 RGB 이미지(227x227)를 입력받는다.
#   2) SqueezeNet 1.1 백본으로 이미지의 특징(feature)을 추출한다.
#   3) 추출된 특징을 1x1 컨볼루션과 FC 레이어로 변환하여
#      num_points 개의 3D 좌표(x, y, z)를 직접 회귀(regression)한다.
#   4) 최종 출력: (batch, num_points, 3) 형태의 텐서.
#
# SqueezeNet을 백본으로 선택한 이유:
#   - 파라미터 수가 매우 적어(약 1.2MB) 학습/추론이 빠르다.
#   - ImageNet으로 사전학습된 가중치를 사용해 적은 데이터로도 좋은 특징을 얻을 수 있다.
#   - Fire 모듈(squeeze -> expand)로 효율적으로 채널을 압축/확장한다.
# =============================================================================

# Python 2/3 호환성을 위한 print 함수 import (레거시 코드 호환 목적)
from __future__ import print_function

# argparse: 커맨드라인 인자 파싱용 (학습 스크립트에서 경로/설정을 받을 때 사용)
import argparse

# os: 파일 경로 조작 등 운영체제 수준의 기능을 사용하기 위해 import
import os

# random: 데이터 셔플 등 재현성 제어용 난수 생성
import random

# torch: PyTorch 메인 패키지 (텐서 연산, 자동 미분 등)
import torch

# torch.nn: 신경망 레이어(Conv, Linear, ReLU 등)를 정의하기 위한 핵심 모듈
import torch.nn as nn

# torch.nn.parallel: 다중 GPU 학습(DataParallel)을 위한 모듈
import torch.nn.parallel

# cudnn: GPU에서 컨볼루션 연산을 가속화하는 백엔드 설정용
import torch.backends.cudnn as cudnn

# optim: 옵티마이저(Adam, SGD 등)로 가중치를 갱신할 때 사용
import torch.optim as optim

# torch.utils.data: DataLoader, Dataset 등 데이터 적재를 위한 유틸리티
import torch.utils.data

# transforms: 이미지 전처리(Resize, ToTensor 등)를 위한 torchvision 모듈
import torchvision.transforms as transforms

# vutils: 이미지를 그리드로 만들어 저장하는 등 시각화 유틸리티
import torchvision.utils as vutils

# Variable: 과거 버전 호환용 (현재는 Tensor 자체로 autograd를 지원하지만 레거시 코드에 남김)
from torch.autograd import Variable

# PIL.Image: 이미지 파일을 열고 변환하기 위한 라이브러리
from PIL import Image

# numpy: 포인트 클라우드(Nx3 배열) 등 수치 데이터 처리용
import numpy as np

# matplotlib: 포인트 클라우드 3D 시각화 등에 사용
import matplotlib.pyplot as plt

# pdb: 디버깅용 파이썬 디버거 (중단점 설정 등)
import pdb

# torch.nn.functional: 함수형 API (활성화 함수, 풀링 등을 함수처럼 호출할 때 사용)
import torch.nn.functional as F

# _Loss, _WeightedLoss: 커스텀 손실 함수(Chamfer Distance 등)를 만들 때 상속받기 위해 import
from torch.nn.modules.loss import _Loss, _WeightedLoss

# math: 수학 함수(제곱근, 로그 등) 사용용
import math

# torch, torch.nn 중복 import는 레거시 코드 잔재 (기능에는 영향 없음)
import torch
import torch.nn as nn

# init: 가중치 초기화 방법(Kaiming 등)을 적용하기 위한 모듈
import torch.nn.init as init

# model_zoo: ImageNet 등으로 사전학습된 가중치를 다운로드/로드하기 위한 유틸리티
import torch.utils.model_zoo as model_zoo


# __all__: 이 모듈에서 외부로 공개할 심볼 목록.
# from pic2points_model import * 시 이 세 가지만 노출되도록 제한한다.
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

# model_urls: SqueezeNet 버전별 ImageNet 사전학습 가중치 다운로드 주소.
# pretrained=True 일 때 이 URL에서 가중치를 자동으로 내려받는다.
model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):
    """
    SqueezeNet의 핵심 블록인 Fire 모듈.

    왜 이 구조를 사용하는가?
      - 1x1 컨볼루션(squeeze)으로 채널 수를 크게 줄여 파라미터를 절약한 뒤,
        1x1과 3x3 컨볼루션(expand)으로 채널을 다시 늘려 표현력을 확보한다.
      - 이 "압축 -> 확장" 패턴은 적은 파라미터로도 풍부한 특징을 학습할 수 있게 해준다.
      - expand 단계에서 1x1과 3x3을 병렬로 수행한 뒤 채널 방향으로 concat하여
        다양한 수용 영역(receptive field)의 정보를 동시에捕获한다.
    """

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        """
        Fire 모듈을 초기화한다.

        왜 각 단계의 채널 수를 분리해서 인자로 받는가?
          - squeeze에서 채널을 얼마나 줄일지, expand에서 1x1/3x3 각각으로
            얼마나 늘릴지를 유연하게 조정하기 위해서다.
          - 일반적으로 squeeze_planes는 inplanes보다 작게 설정하여
            정보 병목(bottleneck) 효과로 파라미터를 줄인다.

        인자:
            inplanes         : 입력 채널 수
            squeeze_planes   : squeeze(1x1) 단계의 출력 채널 수 (압축)
            expand1x1_planes : expand 1x1 단계의 출력 채널 수
            expand3x3_planes : expand 3x3 단계의 출력 채널 수
        """
        super(Fire, self).__init__()

        # 입력 채널 수를 저장 (참조용)
        self.inplanes = inplanes

        # squeeze 단계: 1x1 컨볼루션으로 채널을 inplanes -> squeeze_planes로 압축.
        # 1x1 커널은 공간 정보는 유지한 채 채널 간 선형 결합만 수행하므로 연산량이 적다.
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)

        # squeeze 출력에 ReLU 활성화 함수 적용.
        # inplace=True는 메모리를 절약하기 위해 입력 텐서를 덮어쓰는 옵션이다.
        self.squeeze_activation = nn.ReLU(inplace=True)

        # expand 1x1 단계: squeeze 결과를 1x1 컨볼루션으로 expand1x1_planes 채널로 확장.
        # 1x1은 공간적 패턴은 보지 못하지만 채널 간 조합을 학습한다.
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        # expand 3x3 단계: 3x3 컨볼루션으로 주변 공간 정보를 반영하여 채널을 확장.
        # padding=1은 출력 공간 크기가 입력과 같도록 유지하기 위함이다.
        # (3x3 커널에서 padding=1이면 크기 보존)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        입력 텐서 x를 받아 Fire 블록의 순전파를 수행한다.

        왜 squeeze 후 두 개의 expand를 병렬로 처리하는가?
          - 1x1 expand는 점 단위(채널 간) 정보를, 3x3 expand는 국소적 공간 정보를
            동시에 학습하도록 하여 더 풍부한 표현을 얻기 위해서다.
          - 두 결과를 채널 방향(dim=1)으로 concat하여 다음 레이어에 전달한다.
        """
        # squeeze 단계: 채널 압축 + ReLU
        x = self.squeeze_activation(self.squeeze(x))

        # expand 두 갈래를 각각 활성화한 뒤 채널 차원(dim=1)으로 이어 붙임.
        # 결과 채널 수 = expand1x1_planes + expand3x3_planes
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    """
    SqueezeNet 백본 네트워크 (version 1.0 / 1.1 지원).

    왜 SqueezeNet을 백본으로 사용하는가?
      - AlexNet 수준의 정확도를 파라미터 50배 적게 달성한 경량 모델이다.
      - 1.2MB 수준의 크기로 학습/추론이 빠르고, 메모리 부담이 적다.
      - ImageNet 사전학습 가중치를 사용해 적은 데이터로도 좋은 초기 특징을 얻을 수 있다.

    본 프로젝트에서는 분류용 classifier head를 제거하고 features 부분만 사용하여
    이미지의 특징 맵(feature map)을 추출한다.
    """

    def __init__(self, version=1.0, num_classes=1000):
        """
        SqueezeNet을 초기화한다.

        인자:
            version    : 1.0 또는 1.1 (1.1이 연산량 2.4배 적고 더 효율적)
            num_classes: 분류용 클래스 수 (본 프로젝트에서는 사용하지 않으나
                         호환성을 위해 기본값 1000을 유지한다)

        왜 version을 나누는가?
          - 1.0은 7x7 컨볼루션으로 시작해 연산량이 크고,
            1.1은 3x3 컨볼루션으로 시작해 더 가볍고 효율적이다.
          - 본 프로젝트는 더 가벼운 1.1을 사용한다.
        """
        super(SqueezeNet, self).__init__()

        # 지원하지 않는 버전이면 에러를 발생시켜 잘못된 생성을 방지한다.
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))

        # 분류용 클래스 수 저장 (본 프로젝트에서는 사용하지 않음)
        self.num_classes = num_classes

        if version == 1.0:
            # --- SqueezeNet 1.0 특징 추출부 ---
            # 7x7 컨볼루션으로 시작하여 큰 수용 영역을 빠르게 확보한 뒤,
            # Fire 모듈과 MaxPool을 반복하여 공간 해상도를 줄이고 채널을 늘린다.
            # stride=2 컨볼루션/풀링으로 공간 크기를 절반씩 줄여 연산량을 제어한다.
            self.features = nn.Sequential(
                # 입력 3채널(RGB) -> 96채널, 7x7 커널, stride 2로 공간 축소
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                # 3x3 MaxPool, stride 2로 추가 축소 (ceil_mode=True로 경계 처리)
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # Fire 블록들: 점진적으로 채널을 늘리며 특징을 추출
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            # --- SqueezeNet 1.1 특징 추출부 ---
            # 1.0보다 작은 3x3 컨볼루션으로 시작하여 연산량을 줄였다.
            # Fire 블록 사이사이에 MaxPool을 배치해 공간 해상도를 점진적으로 축소.
            # 최종적으로 512채널의 feature map을 출력한다.
            self.features = nn.Sequential(
                # 입력 3채널(RGB) -> 64채널, 3x3 커널, stride 2 (1.0보다 가벼운 시작)
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        # 모든 모듈을 순회하며 컨볼루션 레이어의 가중치/바이어스를 초기화.
        # 사전학습 가중치를 로드하기 전의 기본 초기화 단계이다.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 바이어스가 있으면 0으로 초기화하여 초기 출력이 치우치지 않게 한다.
                if  m.bias is not None:
                    m.bias.data.zero_()
                else:
                    # 바이어스가 없으면 Kaiming 초기화로 가중치를 설정.
                    # Kaiming 초기화는 ReLU 활성화 함수에 적합하도록
                    # 분산을 조정해 학습 초반 기울기 소실/폭발을 방지한다.
                    init.kaiming_uniform(m.weight.data)

    def forward(self, x):
        """
        입력 이미지 x를 받아 특징 맵(feature map)을 반환한다.

        왜 classifier 없이 features만 반환하는가?
          - 본 프로젝트는 분류가 아닌 3D 포인트 회귀가 목적이므로,
            분류용 classifier head는 제거하고 특징 맵만 다음 단계로 전달한다.
        """
        # features 시퀀스를 통과시켜 특징 맵 추출
        x = self.features(x)
        return x


def squeezenet1_0(pretrained=False, **kwargs):
    """
    SqueezeNet 1.0 모델을 생성하여 반환한다.

    왜 별도의 팩토리 함수를 두는가?
      - 모델 생성과 사전학습 가중치 로드를 한 곳에서 처리해 호출부를 단순화하기 위해서.
      - pretrained=True이면 ImageNet 가중치를 그대로 덮어씌워 초기 특징을 좋게 만든다.

    인자:
        pretrained: True이면 ImageNet 사전학습 가중치를 로드한다.
    """
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # SqueezeNet 1.0 인스턴스 생성
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        # ImageNet 사전학습 가중치를 다운로드하여 모델에 로드.
        # 사전학습을 사용하면 적은 데이터로도 좋은 특징을 빠르게 학습할 수 있다.
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    """
    SqueezeNet 1.1 모델을 생성하여 반환한다.

    왜 1.1을 사용하는가?
      - 1.0 대비 연산량이 2.4배 적고 파라미터도 약간 더 적으면서
        정확도는 거의 희생하지 않는 더 효율적인 버전이기 때문이다.
      - 본 프로젝트의 pic2points 모델이 이 버전을 백본으로 사용한다.

    인자:
        pretrained: True이면 ImageNet 사전학습 가중치를 로드한다.
    """
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # SqueezeNet 1.1 인스턴스 생성
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        # 사전학습 가중치를 다운로드
        pretrained_dict = model_zoo.load_url(model_urls['squeezenet1_1'])
        # 현재 모델의 state_dict를 가져옴
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        # 모델 구조에 없는 키(예: classifier 등)는 제외하여 로드 에러를 방지.
        # 본 프로젝트는 features만 사용하므로 불필요한 가중치는 걸러낸다.
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        # 걸러낸 사전학습 가중치로 현재 모델의 state_dict를 갱신
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        # 갱신된 가중치를 모델에 최종 로드
        model.load_state_dict(pretrained_dict)
    return model


class pic2points(nn.Module):
    """
    단일 이미지를 입력받아 3D 포인트 클라우드(num_points개의 x,y,z 좌표)를
    출력하는 메인 모델.

    왜 이런 구조인가?
      - SqueezeNet 백본으로 이미지의 시각적 특징을 추출한 뒤,
        컨볼루션/FC 레이어로 각 점의 3D 좌표를 직접 회귀한다.
      - 이미지 한 장으로 3D 형태를 복원하는 단일 이미지 -> 3D 재구성 태스크에서
        효율적이고 가벼운 베이스라인으로 동작한다.

    데이터 흐름:
        입력 (3, 227, 227)
          -> input_conv1/2 (1x1 Conv + ReLU)   # 입력 전처리용 경량 변환
          -> SqueezeNet 1.1 features          # (512, 13, 13) 특징 맵
          -> final_conv (512 -> num_points)   # (num_points, 7, 7)
          -> view로 펼침                       # (num_points, 49)
          -> final_fc (49 -> 3)               # (num_points, 3) = 각 점의 x,y,z
    """

    def __init__(self, num_points=2500, batch_size=32):
        """
        pic2points 모델을 초기화한다.

        인자:
            num_points : 생성할 3D 포인트 수 (기본 2500, 본 프로젝트는 11000 사용)
            batch_size : 배치 크기 (참조용으로 저장)

        왜 num_points를 인자로 받는가?
          - 복원하고자 하는 포인트 클라우드의 밀도를 유연하게 조정하기 위해서.
          - 점이 많을수록 형태가 정밀해지지만 연산량과 메모리도 증가한다.
        """
        super(pic2points, self).__init__()

        # 배치 크기 저장 (데이터 로더와의 일관성 유지용)
        self.batch_size = batch_size
        # 생성할 포인트 수 저장 (final_conv 출력 채널 수와 직접 연결됨)
        self.num_points = num_points

        # input_conv1/2: 1x1 컨볼루션으로 입력 이미지 채널(3)을 그대로 3으로 변환.
        # 왜 1x1 컨볼루션을 입력 단에 두 번 넣는가?
        #   - SqueezeNet에 들어가기 전 입력 분포를 모델에 맞게 미세 조정하기 위함이다.
        #   - 1x1 컨볼루션은 공간 구조를 유지하면서 채널 간 선형 결합만 학습하므로
        #     입력 정규화/색상 변환과 유사한 효과를 학습 기반으로 얻을 수 있다.
        self.input_conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.input_conv2 = nn.Conv2d(3, 3, kernel_size=1)

        # random_fully_connected_1/2: 1x1 컨볼루션을 FC처럼 활용하기 위한 레이어.
        # 왜 정의만 하고 forward에서 사용하지 않을 수 있는가?
        #   - 원본 연구의 실험적 설계 잔재로 보이며, 추가 변환 경로를 예비로 둔 것이다.
        #   - 현재 forward에는 사용되지 않으나 구조 호환성을 위해 유지한다.
        self.random_fully_connected_1 = nn.Conv2d(3, 3, kernel_size=1)
        self.random_fully_connected_2 = nn.Conv2d(3, 3, kernel_size=1)

        # SqueezeNet 1.1 백본: ImageNet 사전학습 가중치를 로드하여 사용.
        # 사전학습된 특징 추출기를 가져와 전이학습(transfer learning)의 이점을 얻는다.
        self.squeeze_net = squeezenet1_1(pretrained=True)

        # ReLU 활성화 함수 (inplace로 메모리 절약)
        self.ReLU = nn.ReLU(inplace=True)

        # Final convolution is initialized differently form the rest
        # final_conv: 512채널 특징 맵 -> num_points 채널로 변환.
        # 왜 kernel_size=6인가?
        #   - SqueezeNet 1.1의 출력 feature map이 13x13이고,
        #     6x6 커널(패딩 없음)을 적용하면 7x7 출력이 된다.
        #   - 각 출력 채널이 하나의 포인트에 대응하며, 7x7=49개의 값이
        #     그 포인트의 좌표를 결정하는 입력이 된다.
        self.final_conv = nn.Conv2d(512, self.num_points, kernel_size=6)

        # final_fc: 49(=7x7) 차원 입력 -> 3(x, y, z) 차원 출력.
        # 왜 81이 아니라 49여야 하는가?
        #   - 실제 forward에서 final_conv 출력은 7x7=49이므로 49가 맞으나,
        #     원본 코드는 81로 되어 있어 9x9 출력을 가정한 것으로 보인다.
        #     (코드 로직은 변경하지 않고 원본 그대로 유지한다.)
        #   - 각 포인트별로 49개의 특징 값을 (x, y, z) 3차원 좌표로 압축한다.
        self.final_fc = nn.Linear(81, 3)


    def forward(self, x):
        """
        입력 이미지 x를 받아 (batch, num_points, 3) 포인트 클라우드를 반환한다.

        왜 이 순서로 레이어를 배치하는가?
          - 먼저 입력을 경량 컨볼루션으로 미세 조정하고,
          - SqueezeNet으로 풍부한 특징을 추출한 뒤,
          - final_conv로 "포인트별 채널"을 만들고,
          - final_fc로 각 포인트의 3D 좌표를 회귀한다.
        """
        # 1x1 컨볼루션 + ReLU로 입력 분포를 조정 (SqueezeNet 진입 전 전처리 역할)
        x = self.ReLU(self.input_conv1(x))
        x = self.ReLU(self.input_conv2(x))

        # SqueezeNet 1.1 백본으로 특징 맵 추출 -> (batch, 512, 13, 13)
        x = self.squeeze_net(x)

        # final_conv: 512 -> num_points 채널, 6x6 커널 -> (batch, num_points, 7, 7)
        # ReLU로 비선형성을 추가해 각 포인트의 특징을 활성화
        x = self.ReLU(self.final_conv(x))

        # view: (batch, num_points, 7, 7) -> (batch, num_points, 49)
        # 왜 -1을 사용하는가?
        #   - 남은 차원(7*7=49)을 자동으로 계산하게 하여
        #     공간 차원을 하나로 펼치기 위함이다.
        x = x.view(x.size(0), x.size(1), -1)

        # final_fc: (batch, num_points, 49) -> (batch, num_points, 3)
        # 각 포인트의 특징 벡터를 (x, y, z) 3차원 좌표로 변환(회귀).
        # 최종 출력이 곧 3D 포인트 클라우드가 된다.
        x = self.final_fc(x)

        return x
