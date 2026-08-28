# AGENTS.md — A-Point-Set-Generation (NIA29 3D 물류 포인트 클라우드 재구성)

이 파일은 본 프로젝트를 매번 재분석하지 않아도 되도록 핵심 구조·실행 방법·주의사항을 정리한 참조 문서입니다.
원본 프로젝트: https://arxiv.org/pdf/1612.00603.pdf ("A Point Set Generation Network for 3D Object Reconstruction from a Single Image")

> 본 저장소는 원본(ShapeNet 3D 객체 재구성용)을 **한국 NIA 29차 물류 3D 데이터(단일 이미지 → 3D 포인트 클라우드)** 재구성용으로 개조한 fork입니다. 상품 이미지 한 장을 입력받아 11,000개의 3D 좌표점을 출력하여 물류 박스의 3D 형태를 복원합니다.

AI와의 챗 작업 세션에서 소스 변경이 발생하면 반드시 아래의 파일들도 없데이트 한다.
- AGENTS.md
- README.md
docs/학습및검증절차.md

---

## 1. 프로젝트 개요

- **목표**: 물류 상품의 단일 이미지로부터 3D 포인트 클라우드(11,000개 점)를 생성하여 물품의 3D 형태·부피를 추정.
- **핵심 아이디어**: SqueezeNet 백본으로 이미지 feature를 추출한 후, 1×1 컨볼루션과 FC 레이어를 통해 각 점의 3D 좌표(x, y, z)를 직접 회귀.
- **입력**: 227×227 RGB 상품 이미지 (bbox crop된 PNG).
- **출력**: (N, 3) 포인트 클라우드 — N=11,000개 점의 3D 좌표.
- **손실 함수**: Chamfer Distance (예측 점과 정답 점 사이의 양방향 최근접 거리 평균).
- **평가 지표**: 3D IoU(mIoU), Chamfer Distance loss, 예측 부피 대비 실제 부피.

---

## 2. 디렉토리 구조

```
A-Point-Set-Generation/
├── prepare_data_3d.py           # ★ 전처리 스크립트 (원본 JPG+JSON → PNG+NPY+JSON+txt)
├── baseline_main.py             # 학습+테스트 엔트리포인트 (argparse 경로/설정 지정)
├── train_ajit.py                # ★ 학습 로직 (train/val 함수, 체크포인트 저장, JSON 로그)
├── pic2points_model.py          # ★ 모델 정의 (SqueezeNet 백본 + pic2points 헤드)
├── data_loader.py               # ★ 학습/검증용 데이터 로더 (XDataset, get_loader)
├── data_loader_pix3d.py         # 테스트용 데이터 로더 (TestDataset, 객체 ID 기반)
├── split_data.py                # 데이터 분할 (train/val/test = 8:1:1, 파일명 리스트 저장)
├── datasets.py                  # 원본 PartDataset (ShapeNet용, 참고용 — 현재 미사용)
├── metrics.py                   # ★ 3D IoU, mIoU, 바운딩 박스 부피 계산
├── eval.py                      # ★ 상세 평가 스크립트 (argparse 경로/설정 지정)
├── testing.py                   # 시각화 테스트 (argparse 경로/설정 지정)
├── visualize.py                 # 3D 포인트 클라우드 시각화 유틸
├── Volume_estimation.py         # 부피 추정 스크립트 (argparse 경로/설정 지정, max1/max2 버그 수정됨)
├── chamfer_distance.py          # ★ 순수 PyTorch Chamfer Distance (pytorch3d 대체, Windows용)
├── requirements.txt             # ★ Linux 서버용 패키지 목록 (pip install -r)
├── .gitignore                   # Git 제외 설정 (.venv/, *.pt, datasets/ 등)
├── datasets/                    # ★ 전처리 출력 (prepare_data_3d.py가 생성)
│   ├── images_poly_bbox_crop/   #   폴리곤 crop PNG
│   ├── labels/npy_stride5/      #   박스 포인트 클라우드 NPY (11000×3)
│   ├── json_labels/             #   치수 JSON
│   ├── train_data.txt           #   학습 파일명 리스트
│   ├── val_data.txt             #   검증 파일명 리스트
│   └── test_data.txt            #   테스트 파일명 리스트
├── Baseline_DL_Vis              # 학습 로그 JSON (epoch별 loss/mIoU 기록)
├── Baseline_DL_Vis.pt           # 최신 체크포인트 (전체 모델 저장)
├── best-Baseline_DL_Vis.pt      # ★ best 검증 loss 체크포인트
├── train_data.txt               # 학습 데이터 파일명 리스트 (전처리 후 datasets/로 이동)
├── val_data.txt                 # 검증 데이터 파일명 리스트
├── test_data.txt                # 테스트 데이터 파일명 리스트
├── box_volumes.csv              # 부피 추정 결과 (Volume_estimation.py 출력)
├── pytorch3d/                   # PyTorch3D 라이브러리 (Chamfer Distance용, 서드파티, Linux 빌드)
├── PT/                          # (빈 디렉토리, 체크포인트 보관용)
├── logs/                        # 평가 로그 (eval_YYYYMMDD_HHMMSS.log)
├── eval_npy/                    # 평가 시 예측 포인트 클라우드 npy 저장
├── result/                      # 시각화 결과 이미지
└── README.md                    # 원본 설명
```

---

## 3. 데이터 형식 (NIA29 3D)

### 3.1 디렉토리 레이아웃

**전처리 후 데이터** (`prepare_data_3d.py`가 생성, 모든 스크립트의 기본 경로):
```
datasets/
  images_poly_bbox_crop/          ← baseline_main.py --image_root (기본값)
    └── <파일명>.png               #   폴리곤 crop된 상품 이미지 (PNG)
  labels/                         ← baseline_main.py --point_cloud_root (기본값)
    └── npy_stride5/<파일명>.npy   #   N×3 numpy 배열 (x, y, z 좌표)
  json_labels/                    ← Volume_estimation.py --json_root (기본값)
    └── <파일명>.json              #   COCO-like, attributes에 width/length/height 포함
  train_data.txt                  ← baseline_main.py --train_list (기본값)
  val_data.txt                    ← baseline_main.py --val_list (기본값)
  test_data.txt                   ← baseline_main.py --test_list (기본값)
```

> **경로 지정**: 모든 스크립트가 argparse로 경로를 받는다. 상대경로/절대경로 모두 가능.
> 기본값은 `datasets/` 하위 경로이므로, 전처리 후 별도 수정 없이 실행 가능.
> ```bash
> python baseline_main.py --image_root /data/img --point_cloud_root /data/pcd
> ```

### 3.2 데이터 분할 (`split_data.py`)

- 분할 비율: train 80%, val 10%, test 10%
- 분할 기준: 파일명 단위 무작위 분할 (이미지 파일명만 저장)
- 랜덤 시드: 169 (고정)
- 결과: `train_data.txt`, `val_data.txt`, `test_data.txt` (CSV 형식, 파일명만 저장)
- 이미 파일이 존재하면 덮어쓰지 않음 (`overrideFiles=False`)

### 3.3 데이터 로더 (`data_loader.py`)

**XDataset** (학습/검증용):
- 입력: 이미지 루트, 포인트 클라우드 루트, 파일명 리스트
- 초기화 시 `os.walk`로 모든 하위 디렉토리를 순회하여 이미지-포인트클라우드 쌍을 미리 매핑 (속도 향상)
- `__getitem__`: 이미지는 PIL → transform → tensor, 포인트 클라우드는 np.load → numpy 배열
- 반환: (image_tensor, point_cloud_numpy)

**TestDataset** (`data_loader_pix3d.py`, 테스트용):
- 객체 ID 리스트(`B120110053` 등)를 입력받아 이미지-포인트클라우드 쌍을 찾음
- 객체 ID는 박스 크기 코드 (예: B120110053 = 가로120×세로110×높이53)

### 3.4 이미지 전처리

```python
transform = transforms.Compose([
    transforms.Resize(227, interpolation=2),   # 227×227 (SqueezeNet 입력 크기)
    transforms.CenterCrop(227),
    transforms.ToTensor(),                      # [0, 1] 정규화
])
```

> **주의**: ImageNet 표준 정규화(mean/std)를 사용하지 않고 ToTensor만 적용한다. 이는 원본 코드의 설계 선택이다.

---

## 4. 모델 아키텍처 (`pic2points_model.py`)

### 4.1 전체 구조: `pic2points`

```
입력 이미지 (3, 227, 227)
  → input_conv1 (1×1 Conv) → ReLU
  → input_conv2 (1×1 Conv) → ReLU
  → SqueezeNet 1.1 (ImageNet 사전학습)
    → features: (512, 13, 13)
  → final_conv (512→num_points, 6×6 Conv)
    → (num_points, 7, 7)
  → view: (num_points, 49)
  → final_fc (49→3)
    → (num_points, 3)  ← 각 점의 (x, y, z) 좌표
```

### 4.2 SqueezeNet 1.1 백본

- ImageNet 사전학습 가중치 로드 (`squeezenet1_1(pretrained=True)`)
- 가벼운 모델 (1.2MB), AlexNet 수준 정확도 with 50배 적은 파라미터
- Fire 모듈: squeeze(1×1) → expand(1×1 + 3×3) → concat
- 출력: 512채널 feature map

### 4.3 포인트 생성 헤드

- `final_conv`: 512채널 → num_points(11000)채널, 6×6 커널
  - 13×13 feature map → 7×7 → 각 채널이 하나의 점에 대응
- `final_fc`: 49(7×7) → 3 (x, y, z 좌표)
- 최종 출력: (batch_size, 11000, 3)

### 4.4 핵심 파라미터

| 파라미터 | 기본값 | 비고 |
|---|---|---|
| `num_points` | 11000 | 생성할 포인트 수 (`--num_points`) |
| `img_size` | 227 | SqueezeNet 입력 크기 (변경 불가) |
| `batch_size` | 32 | 학습 시 (`--batch_size`) |
| `num_workers` | 8 | DataLoader 워커 수 (`--num_workers`, Windows: 0 권장) |
| `learning_rate` | 0.001 | Adam 학습률 (`--learning_rate`) |
| `num_epochs` | 50 | 총 학습 epoch (`--num_epochs`) |

---

## 5. 학습 파이프라인 (`baseline_main.py` + `train_ajit.py`)

### 5.1 실행 흐름

1. `baseline_main.py`가 argparse로 인자 파싱 (경로, 학습 설정, 모델 이름 등)
2. `split_data()`로 train/val/test 분할 (이미 분할되어 있으면 스킵)
3. `get_loader()`로 데이터 로더 생성
4. `pic2points(num_points=11000)` 모델 생성
5. 다중 GPU 시 `DataParallel` 적용
6. `train()` 함수 호출 (train_ajit.py)

### 5.1.1 baseline_main.py argparse 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--training` | `True` | 학습 모드 여부 (True/False) |
| `--image_root` | `datasets/images_poly_bbox_crop` | 이미지 루트 경로 |
| `--point_cloud_root` | `datasets/labels` | 포인트 클라우드 루트 경로 |
| `--train_list` | `train_data.txt` | 학습 리스트 파일 |
| `--val_list` | `val_data.txt` | 검증 리스트 파일 |
| `--test_list` | `test_data.txt` | 테스트 리스트 파일 |
| `--model_name` | `Baseline_DL_Vis` | 모델 저장 이름 |
| `--num_epochs` | `50` | epoch 수 |
| `--batch_size` | `32` | 배치 크기 |
| `--num_workers` | `8` | 워커 수 (Windows: 0 권장) |
| `--learning_rate` | `0.001` | 학습률 |
| `--num_points` | `11000` | 포인트 수 |
| `--use_checkpoint` | (플래그) | 체크포인트에서 학습 재개 |

```bash
# 기본 실행 (datasets/ 경로 사용)
python baseline_main.py --training True

# Windows 디버깅
python baseline_main.py --batch_size 4 --num_workers 0

# 커스텀 경로
python baseline_main.py --image_root /data/img --point_cloud_root /data/pcd
```

### 5.2 학습 로직 (`train_ajit.py`의 `train()`)

- **옵티마이저**: Adam (lr=0.001)
- **손실 함수**: Chamfer Distance
  - `dist1`: 정답 점 → 예측 점 최근접 거리
  - `dist2`: 예측 점 → 정답 점 최근접 거리
  - `loss = (mean(dist1) + mean(dist2)) / 100.0` (스케일 정규화)
- **평가 지표**: 3D mIoU (metrics.py의 `calculate_3d_miou`)
- **체크포인트 저장**:
  - 매 epoch: `{model_name}.pt` (전체 모델)
  - best val_loss: `best-{model_name}.pt`
  - 학습 로그: `{model_name}` (JSON 파일, epoch별 loss/mIoU/time)
- **학습 재개**: `use_checkpoint=True` 시 JSON 로그에서 진행된 epoch 수를 읽어 이어서 학습

### 5.3 검증 (`train_ajit.py`의 `val()`)

- `model.eval()` + `torch.no_grad()`
- Chamfer Distance loss + 3D mIoU 계산
- 검증 loss가 최소일 때 best 모델로 저장

### 5.4 학습 로그 형식 (`Baseline_DL_Vis` JSON)

```json
[
  {
    "training_loss": 0.380,
    "training_miou": 0.180,
    "val_loss": 0.180,
    "val_miou": 0.273,
    "epoch": 1,
    "time": "0:07:20"
  },
  ...
]
```

### 5.5 주의: 두 개의 train 버전

`train_ajit.py`에는 두 개의 `train()` 함수가 있다:
- **활성 코드 (1~139줄)**: 단순 체크포인트 저장 (전체 모델 torch.save)
- **주석 처리된 코드 (187~425줄)**: 더 체계적인 버전 (PT&logs/ 디렉토리, 5에폭마다 체크포인트, state_dict 저장)

> 현재 활성 코드는 전체 모델을 `torch.save(model)`로 저장하여 로드 시 모델 클래스 정의가 필요하다.

---

## 6. 평가 스크립트 (`eval.py`)

### 6.1 기능

- 학습된 모델로 테스트 데이터셋 평가
- 이미지별 예측 포인트 클라우드를 `.npy` 파일로 저장 (`eval_npy/`)
- 이미지별 IoU, 예측 부피, 정답 부피를 로그에 출력
- 최종 평균 IoU 계산

### 6.1.1 eval.py argparse 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--image_root` | `datasets/images_poly_bbox_crop` | 이미지 루트 경로 |
| `--point_cloud_root` | `datasets/labels` | 포인트 클라우드 루트 경로 |
| `--model_path` | `best-Baseline_DL_Vis.pt` | 모델 체크포인트 경로 |
| `--test_list` | `test_data.txt` | 테스트 리스트 파일 |
| `--save_dir` | `./eval_npy` | 예측 NPY 저장 디렉토리 |
| `--batch_size` | `32` | 배치 크기 |
| `--num_workers` | `8` | 워커 수 (Windows: 0 권장) |
| `--num_points` | `11000` | 포인트 수 |

```bash
# 기본 실행
python eval.py

# 커스텀 경로
python eval.py --image_root /data/img --model_path /models/best.pt
```

### 6.2 로그 출력

```
Item : 01010101_8801039920121, pred_vol: 22141.28, gt_vol: 16800.00, IOU: 0.7573
```

- `Item`: 상품 ID (KAN_code + 바코드)
- `pred_vol`: 예측 포인트 클라우드의 바운딩 박스 부피
- `gt_vol`: 정답 포인트 클라우드의 바운딩 박스 부피
- `IOU`: 3D IoU (0~1, 높을수록 좋음)

### 6.3 로그 위치

- `logs/eval_YYYYMMDD_HHMMSS.log`

---

## 7. 시각화 (`testing.py` + `visualize.py`)

### 7.1 `testing.py`

- 테스트 데이터에 대해 예측 포인트 클라우드를 3D scatter plot으로 시각화
- 원본 이미지와 포인트 클라우드를 나란히 배치하여 PNG로 저장
- 저장 위치: `result/visual/` (자동으로 번호 증가)

### 7.1.1 testing.py argparse 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--image_root` | `datasets/images_poly_bbox_crop` | 이미지 루트 경로 |
| `--point_cloud_root` | `datasets/labels` | 포인트 클라우드 루트 경로 |
| `--model_path` | `best-Baseline_DL_Vis.pt` | 모델 체크포인트 경로 |
| `--test_list` | `test_data.txt` | 테스트 리스트 파일 |
| `--save_dir` | `./result/visual` | 시각화 결과 저장 디렉토리 |
| `--num_workers` | `8` | 워커 수 (Windows: 0 권장) |
| `--visualize_points` | `3000` | 시각화 포인트 수 |

```bash
# 기본 실행
python testing.py

# 커스텀 경로
python testing.py --model_path /models/best.pt --save_dir ./output/visual
```

### 7.2 `visualize.py`의 `Visualize` 클래스

- `ShowResult(img_tensor, save_path)`: 원본 이미지 + 3D 포인트 클라우드 시각화
- z값에 따라 viridis 컬러맵 적용
- 3000개 점만 무작위 샘플링하여 표시 (성능 최적화)

---

## 8. 부피 추정 (`Volume_estimation.py`)

- 예측 포인트 클라우드로부터 박스 치수(가로/세로/높이)를 추정
- JSON 라벨의 실제 치수(width/length/height)와 비교
- 스케일 보정: 예측 치수의 평균과 실제 치수의 평균 비율로 스케일 계산
- 결과를 `box_volumes.csv`로 저장

### 8.1 Volume_estimation.py argparse 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--image_root` | `datasets/images_poly_bbox_crop` | 이미지 루트 경로 |
| `--point_cloud_root` | `datasets/labels` | 포인트 클라우드 루트 경로 |
| `--json_root` | `datasets/json_labels` | 치수 JSON 루트 경로 |
| `--model_path` | `best-Baseline_DL_Vis.pt` | 모델 체크포인트 경로 |
| `--test_list` | `test_data.txt` | 테스트 리스트 파일 |
| `--num_workers` | `8` | 워커 수 (Windows: 0 권장) |
| `--num_points` | `11000` | 포인트 수 |
| `--csv_path` | `box_volumes.csv` | 결과 CSV 저장 경로 |

```bash
# 기본 실행
python Volume_estimation.py

# 커스텀 경로
python Volume_estimation.py --json_root /data/json --model_path /models/best.pt
```

> **버그 수정 완료**: 기존 114줄의 `max1`/`max2` 미정의 변수 참조 오류를
> `mean1`/`mean2`로 수정하여 해결.

---

## 9. 평가 지표 (`metrics.py`)

### 9.1 3D IoU 계산

1. **정규화**: GT 포인트 클라우드의 바운딩 박스를 기준으로 pred와 GT를 [0, 1] 공간으로 정규화
2. **바운딩 박스**: 각 포인트 클라우드의 min/max 좌표로 AABB 계산
3. **부피**: AABB 부피 = (x_max - x_min) × (y_max - y_min) × (z_max - z_min)
4. **교차 부피**: 두 AABB의 겹치는 영역 부피
5. **IoU**: 교차 부피 / 합집합 부피

### 9.2 mIoU

- 배치 내 모든 샘플의 IoU 평균

### 9.3 주의: 두 개 버전

- **활성 코드 (1~119줄)**: 정규화 기반 IoU (GT 기준 정규화 후 계산)
- **주석 처리 (121~174줄)**: 비정규화 IoU (원본 좌표 그대로 계산)
- **주석 처리 (176~221줄)**: Voxel 기반 IoU (8×8×8 복셀화 후 계산)

---

## 10. Chamfer Distance

### 10.1 두 가지 구현 방식

본 프로젝트는 환경에 따라 두 가지 Chamfer Distance 구현을 사용:

1. **Linux 서버 (권장)**: `pytorch3d` 라이브러리에서 제공 (`from chamfer_distance import ChamferDistance`)
   - `pytorch3d/` 디렉토리에 전체 라이브러리가 포함되어 있음
   - CUDA 가속된 C++/CUDA 커널로 고속 처리
   - 소스 빌드 필요: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`

2. **Windows / pytorch3d 미설치 환경**: 저장소 내 `chamfer_distance.py` (순수 PyTorch 구현)
   - pytorch3d가 설치되어 있지 않으면 자동으로 이 파일이 import됨
   - 컴파일 불필요, CPU/GPU 모두 지원
   - 11000점 포인트 클라우드에 대한 청크 단위 메모리 최적화
   - 동일 API: `from chamfer_distance import ChamferDistance`

### 10.2 사용법 (공통)

```python
chamferDist = ChamferDistance()
dist1, dist2 = chamferDist(pred, gt)  # dist1: gt→pred, dist2: pred→gt
loss = (torch.mean(dist1) + torch.mean(dist2)) / 100.0
```

### 10.3 chamfer_distance.py (순수 PyTorch 구현)

- 파일 위치: 저장소 루트 `chamfer_distance.py`
- pytorch3d가 설치된 경우 pytorch3d가 우선 적용됨 (이 파일은 무시됨)
- 11000×11000 거리 행렬을 청크 단위(2048)로 분할하여 메모리 절약
- 제곱 거리(squared distance) 반환 — pytorch3d와 동일한 동작

---

## 11. 환경 요구사항

### 11.1 필수 패키지

`requirements.txt`에 정의됨. 설치 방법:

```bash
# 1. 가상환경 생성 및 활성화 (Linux)
python -m venv .venv
source .venv/bin/activate

# 2. PyTorch CUDA 설치 (GPU에 맞는 CUDA 버전 선택)
# RTX 50xx (Blackwell):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# RTX 30xx/40xx:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 나머지 패키지 설치
pip install -r requirements.txt

# 4. pytorch3d 빌드 (Linux 권장, Windows에서는 chamfer_distance.py 사용)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

| 패키지 | 용도 | 비고 |
|--------|------|------|
| torch, torchvision | 학습/추론 | CUDA 빌드 별도 설치 |
| numpy | 포인트 클라우드 | |
| pillow | 이미지 로드 | |
| tqdm | 진행률 | |
| matplotlib | 시각화 | testing.py, visualize.py |
| pandas | 부피 추정 | Volume_estimation.py |
| scikit-learn | 평가 | eval.py |
| opencv-python | datasets.py import | 현재 미사용이나 import 필요 |
| imageio | data_loader_pix3d.py | |
| open3d | Volume_estimation.py | Python 3.14 미지원, 현재 미사용 (try/except 처리) |
| pytorch3d | Chamfer Distance | Linux 빌드 권장, Windows는 chamfer_distance.py 대체 |
| ninja | pytorch3d 빌드 도구 | |

### 11.2 GPU

- 학습: GPU 필수 (RTX 2080 Ti × 8 환경에서 약 7분/epoch)
- 평가/시각화: GPU 권장, CPU도 가능

### 11.3 Windows 주의사항

- 모든 스크립트가 argparse로 경로를 받으므로 코드 수정 불필요
- `--num_workers 0`으로 설정 (Windows 다중 워커 문제)
- `--batch_size 4~8`로 축소 (GPU 메모리에 따라)
- pytorch3d 대신 저장소 내 `chamfer_distance.py` 사용 (별도 빌드 불필요)
- `torch.load()`에 `map_location`이 추가되어 CPU 환경에서도 로드 가능

---

## 12. 알려진 이슈 / 코드 특이사항

1. **경로 하드코딩 해결됨**: 모든 스크립트가 argparse로 경로를 받는다. 기본값은 `datasets/` 하위 경로. 상대경로/절대경로 모두 지원.

2. **두 개 버전 문제**: `train_ajit.py`, `data_loader.py`, `metrics.py`, `eval.py`, `testing.py`, `visualize.py` 모두에 주석 처리된 이전 버전이 존재. 활성 코드는 위쪽.

3. **전체 모델 저장**: `torch.save(model)`로 전체 모델을 저장하므로, 로드 시 `pic2points` 클래스 정의가 필요 (`torch.load` 시 클래스가 import되어 있어야 함). `map_location`이 추가되어 CPU/GPU 모두 로드 가능.

4. **Volume_estimation.py 버그 수정됨**: 기존 114줄의 `max1`/`max2` 미정의 변수를 `mean1`/`mean2`로 수정하여 해결.

5. **split_data.py 확장자**: `.png` 파일만 검색하므로, 데이터가 JPG인 경우 수정 필요. 단, `prepare_data_3d.py`로 전처리 시 PNG로 변환되므로 전처리 후에는 문제 없음.

6. **SqueezeNet 입력 크기**: `img_size=227`로 고정. SqueezeNet 아키텍처상 변경 불가.

7. **num_points=11000**: 매우 큰 포인트 수. 메모리와 연산량이 크므로 GPU 메모리에 따라 batch_size 조절 필요. `--num_points`로 변경 시 `prepare_data_3d.py --num_points`와 일치해야 함.

8. **데이터 로더 인덱싱**: `XDataset` 초기화 시 `os.walk`로 전체 디렉토리를 순회하여 파일을 찾는다. 데이터가 많으면 초기화 시간이 길 수 있음.

9. **eval.py의 img_names 처리**: `val_data[i * batch_size:(i + 1) * batch_size]`로 파일명을 가져오므로, 마지막 배치가 batch_size보다 작을 수 있음 (drop_last=False).

10. **학습 로그 JSON 파일명**: 모델 이름과 동일한 파일명(`Baseline_DL_Vis`)에 JSON 로그를 저장. 확장자가 없으므로 체크포인트(.pt)와 혼동 주의.

---

## 13. 자체 데이터로 학습시키기 위한 체크리스트

새 데이터를 본 프레임워크로 학습하려면:

1. **전처리 실행**: `prepare_data_3d.py --src <원본경로> --dst datasets --clean`으로 전처리. PNG, NPY, JSON, txt 파일이 자동 생성됨.
2. **데이터 경로 지정**: 모든 스크립트가 argparse로 경로를 받으므로 코드 수정 불필요. `--image_root`, `--point_cloud_root` 등으로 지정.
3. **이미지 형식**: 전처리 후 PNG 파일이 생성됨. 원본이 JPG여도 전처리 시 PNG로 변환.
4. **포인트 클라우드 형식**: `.npy` 파일 (N×3 numpy 배열). 파일명은 이미지 파일명과 동일(확장자만 다름). 전처리 시 자동 생성.
5. **데이터 분할**: `prepare_data_3d.py`가 train/val/test txt 파일을 자동 생성. `split_data.py` 실행 불필요.
6. **모델 경로 지정**: `eval.py`, `testing.py`, `Volume_estimation.py`의 `--model_path`로 지정.
7. **num_points 조정**: `prepare_data_3d.py --num_points N`과 모델 `--num_points N`을 일치시켜야 함.
8. **Windows 설정**: `--num_workers 0`, `--batch_size 4~8` 권장.
9. **Chamfer Distance**: Linux에서는 pytorch3d 빌드 권장, Windows에서는 `chamfer_distance.py`(순수 PyTorch) 자동 사용.

---

## 14. 빠른 참조: 파일별 핵심 라인

- `prepare_data_3d.py`
  - `scan_products()`: 원본 데이터 스캔
  - `crop_image_with_polygon()`: 폴리곤 crop
  - `generate_box_point_cloud()`: 박스 포인트 클라우드 생성
  - `split_products()`: 상품 단위 분할
  - argparse: `--src`, `--dst`, `--category`, `--num_points`, `--clean`, `--dry_run`
- `chamfer_distance.py` (순수 PyTorch Chamfer Distance, Windows용)
  - `ChamferDistance` 클래스: pytorch3d와 동일 API
  - `_chamfer_distance()`: 청크 단위 메모리 최적화 (chunk_size=2048)
  - `_squared_distance_matrix()`: ||x||² + ||y||² - 2xy 공식
  - pytorch3d 설치 시 pytorch3d가 우선 적용됨
- `requirements.txt` (Linux 서버용 패키지 목록)
  - torch/torchvision: 별도 CUDA 인덱스 URL로 설치
  - pytorch3d: 소스 빌드 권장 (주석에 설치 명령어 포함)
  - open3d: Python 3.14 미지원, 주석 처리 (현재 미사용)
  - `scan_products()`: 원본 데이터 스캔
  - `crop_image_with_polygon()`: 폴리곤 crop
  - `generate_box_point_cloud()`: 박스 포인트 클라우드 생성
  - `split_products()`: 상품 단위 분할
  - argparse: `--src`, `--dst`, `--category`, `--num_points`, `--clean`, `--dry_run`
- `baseline_main.py`
  - argparse 인자 파싱: 38~70줄
  - 데이터 경로: `--image_root`, `--point_cloud_root` (기본값: datasets/ 하위)
  - 학습 설정: `--num_epochs`, `--batch_size`, `--num_workers`, `--learning_rate`, `--num_points`
  - 모델 생성: `pic2points(num_points=num_points)`
  - train 호출: `train(model, ..., model_name=args.model_name, ...)`
  - 테스트 (Pix3D): image_root, point_cloud_root 재사용
- `train_ajit.py`
  - `train()`: 33~139줄 (활성 코드)
  - `val()`: 141~173줄
  - 체크포인트 저장: 123~126줄
  - JSON 로그: 128~137줄
  - 주석 처리된 개선 버전: 187~425줄
- `pic2points_model.py`
  - SqueezeNet 정의: 56~106줄
  - `pic2points` 클래스: 142~166줄
  - `forward()`: 158~166줄
- `data_loader.py`
  - `XDataset` (활성): 75~142줄
  - `get_loader()`: 144~155줄
  - 주석 처리된 이전 버전: 11~71줄
- `metrics.py`
  - `calculate_3d_iou()`: 57~91줄 (활성, 정규화 기반)
  - `calculate_3d_miou()`: 93~105줄
  - `normalize_points()`: 3~22줄
- `eval.py`
  - argparse 인자 파싱: main() 내
  - `save_prediction_npy()`: 69~94줄
  - 데이터 경로: `--image_root`, `--point_cloud_root`, `--model_path`
  - 모델 로드: `torch.load(model_path, map_location=device)`
- `testing.py`
  - argparse 인자 파싱: main() 내
  - 데이터 경로: `--image_root`, `--point_cloud_root`, `--model_path`
  - 모델 로드: `torch.load(model_path, map_location=gpu_or_cpu)`
- `split_data.py`
  - `split_data()`: 20~49줄
  - 이미지 루트: 28줄 (하드코딩 — 전처리 후에는 사용 불필요)
  - 랜덤 시드: 39줄
- `Volume_estimation.py`
  - argparse 인자 파싱: main() 내
  - `find_json_data()`: 23~36줄
  - 부피 계산: 107~118줄
  - 스케일 계산: `mean1 / mean2` (버그 수정됨)
  - 데이터 경로: `--image_root`, `--point_cloud_root`, `--json_root`, `--model_path`
