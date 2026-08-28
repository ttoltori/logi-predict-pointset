#!/usr/bin/env python3
"""
prepare_data_3d.py
==================
NIA29 02_출고물품 원천데이터(JPG + JSON)를 A-Point-Set-Generation 학습용으로 전처리.

이 스크립트가 하는 일:
  1. 원천 이미지를 폴리곤 segmentation 기반으로 crop하여 PNG로 저장
  2. JSON의 박스 치수(length, width, height)로부터 3D 박스 포인트 클라우드 생성 → NPY 저장
  3. 치수 정보만 추출한 간소화 JSON 저장 (Volume_estimation.py용)
  4. 상품 단위 stratified 분할 → train/val/test txt 파일 생성

왜 이 전처리가 필요한가:
  - A-Point-Set-Generation 모델은 "단일 이미지 → 3D 포인트 클라우드"를 학습
  - 원본 NIA29 데이터에는 3D 스캔 포인트 클라우드가 없음
  - 대신 JSON에 상품의 물리적 치수(length/width/height)가 있으므로
    박스 형태의 포인트 클라우드를 합성하여 정답(GT)으로 사용
  - 이는 "물류공간 예측" 목적에 부합 — 상품이 차지하는 공간을 박스로 근사

입력 구조:
  <src>/
    01.원천데이터/02_출고물품/<대분류>/<중분류>/<KAN>_<id>/
      <KAN>_<id>_<shot>_<cam>.jpg
    02.라벨링데이터/02_출고물품/<대분류>/<중분류>/<KAN>_<id>/
      <KAN>_<id>_<shot>_<cam>.json

출력 구조:
  <dst>/
    images_poly_bbox_crop/          ← 폴리곤 bbox crop PNG (data_loader.py의 image_root)
      <KAN>_<id>_<shot>_<cam>.png
    labels/npy_stride5/             ← 박스 포인트 클라우드 NPY (data_loader.py의 point_cloud_root)
      <KAN>_<id>_<shot>_<cam>.npy
    json_labels/                    ← 치수 JSON (Volume_estimation.py의 json_root)
      <KAN>_<id>_<shot>_<cam>.json
    train_data.txt                  ← 학습 파일명 리스트 (split_data.py 호환)
    val_data.txt                    ← 검증 파일명 리스트
    test_data.txt                   ← 테스트 파일명 리스트

사용법:
  python prepare_data_3d.py                                          # 기본값
  python prepare_data_3d.py --src ../Sample --dst datasets           # 경로 지정
  python prepare_data_3d.py --category outbound --num_points 11000   # 출고물품, 11000점
  python prepare_data_3d.py --clean                                  # 기존 출력 삭제 후 재생성
  python prepare_data_3d.py --dry_run                                # 전처리 없이 통계만 출력
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Windows 콘솔(cp949)에서 한글/특수문자 출력 보장
# 왜 필요한가: Windows 기본 인코딩이 cp949라서 한글 로그가 깨질 수 있음
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ── 상수 ──────────────────────────────────────────────────────────

# samples 내 원천/라벨링 폴더명
RAW_DIR = '01.원천데이터'
ANN_DIR = '02.라벨링데이터'

# 물품 카테고리 별칭 → 실제 폴더명 매핑
# 왜 별칭을 쓰는가: 사용자가 "inbound"/"outbound"로 직관적으로 지정할 수 있도록
CATEGORY_MAP = {
    'inbound': '01_입고물품',
    'outbound': '02_출고물품',
}
CATEGORY_DEFAULT = 'outbound'  # 본 스크립트의 주 목적이 출고물품이므로 기본값을 outbound로 설정

# 출력 하위 디렉토리명 (A-Point-Set-Generation 코드가 기대하는 이름)
IMG_DIR = 'images_poly_bbox_crop'
PCD_DIR = 'labels'
PCD_SUBDIR = 'npy_stride5'
JSON_DIR = 'json_labels'

# 포인트 클라우드 생성 파라미터
DEFAULT_STRIDE = 0.5       # cm 단위 (5mm 간격, "stride5"와 일치)
DEFAULT_NUM_POINTS = 11000 # 모델의 num_points와 일치 (DataLoader 배치 처리를 위해 고정 필요)


# ── 데이터 스캔 ────────────────────────────────────────────────────

def scan_products(src: Path, category: str = CATEGORY_DEFAULT):
    """
    원천데이터와 라벨링데이터에서 지정 카테고리의 상품(leaf 폴더) 단위로
    이미지·어노테이션 파일 쌍을 수집.

    왜 상품(leaf 폴더) 단위로 수집하는가:
      - 동일 상품의 여러 이미지(여러 각도/카메라)가 train/val/test에 섞이면
        데이터 누수(data leakage)가 발생하여 평가 성능이 과대평가됨
      - 상품 단위 분할은 실제 서비스에서 새 상품을 처리하는 상황을 시뮬레이션

    Args:
        src: 원본 데이터 루트 (Sample/ 등)
        category: 'inbound' | 'outbound' | 실제 폴더명

    Returns:
        products: list of dict, 각 상품:
          {
            'kan_code': '0101',              # KAN_code 앞 4자리
            'product_folder': '01010102_...', # leaf 폴더명
            'pairs': [(img_path, json_path), ...]  # 이미지-JSON 쌍 리스트
          }
    """
    category_folder = CATEGORY_MAP.get(category, category)

    raw_root = src / RAW_DIR
    ann_root = src / ANN_DIR

    if not raw_root.exists():
        print(f'[오류] 원천데이터 디렉토리가 없습니다: {raw_root}')
        sys.exit(1)
    if not ann_root.exists():
        print(f'[오류] 라벨링데이터 디렉토리가 없습니다: {ann_root}')
        sys.exit(1)

    raw_cat = raw_root / category_folder
    ann_cat = ann_root / category_folder
    if not raw_cat.exists():
        print(f'[오류] 카테고리 폴더가 원천데이터에 없습니다: {raw_cat}')
        print(f'  사용 가능: {list(CATEGORY_MAP.keys())} 또는 {list(CATEGORY_MAP.values())}')
        sys.exit(1)
    if not ann_cat.exists():
        print(f'[오류] 카테고리 폴더가 라벨링데이터에 없습니다: {ann_cat}')
        sys.exit(1)

    # 이미지가 들어있는 leaf 폴더(=상품 폴더) 수집
    product_folders = []
    for dirpath, _, filenames in os.walk(raw_cat):
        jpgs = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg'))]
        if jpgs:
            product_folders.append(Path(dirpath))

    products = []
    for pf in product_folders:
        folder_name = pf.name  # 예: 01010102_8809162740524
        kan_code = folder_name[:4]

        # 라벨링 데이터에서 대응하는 폴더 찾기 (동일한 상대경로)
        rel = pf.relative_to(raw_cat)
        ann_pf = ann_cat / rel

        # 이미지-JSON 쌍 매칭
        # 왜 직접 매칭하는가: 파일명이 동일하므로(확장자만 다름) 명시적으로 짝을 지어야 함
        pairs = []
        for img_file in sorted(pf.glob('*.jp*g')):
            json_name = img_file.stem + '.json'
            json_path = ann_pf / json_name
            if json_path.exists():
                pairs.append((img_file, json_path))
            else:
                print(f'  [경고] JSON 누락: {json_path.name} (이미지만 있음)')

        if pairs:
            products.append({
                'kan_code': kan_code,
                'product_folder': folder_name,
                'pairs': pairs,
            })

    return products


# ── 이미지 crop ────────────────────────────────────────────────────

def crop_image_with_polygon(img_path: Path, json_data: dict, out_path: Path):
    """
    폴리곤 segmentation과 bbox를 이용하여 이미지에서 상품 영역만 crop.

    왜 폴리곤 crop을 하는가:
      - 원본 이미지(1920x1080)에는 배경이 포함되어 있어 모델이 상품 영역에 집중하기 어려움
      - 폴리곤 마스크로 상품 영역만 남기고 배경을 흰색으로 채우면
        모델이 상품의 형태를 더 쉽게 학습할 수 있음
      - "poly_bbox_crop"이라는 출력 디렉토리명이 이 전처리를 의미

    처리 순서:
      1. bbox 영역만 잘라내기
      2. 폴리곤 외부 영역을 흰색(255,255,255)으로 채우기
      3. PNG로 저장 (무손실 압축, 투명도 지원)

    Args:
        img_path: 원본 JPG 경로
        json_data: JSON 라벨 딕셔너리
        out_path: 출력 PNG 경로
    """
    # 어노테이션에서 bbox와 segmentation 추출
    ann = json_data['annotations'][0]
    bbox = ann['bbox']  # [x, y, width, height]
    segmentation = ann['segmentation']  # [[x1, y1, x2, y2, ...], ...]

    # bbox를 정수로 변환 (픽셀 인덱스로 사용)
    bx, by, bw, bh = [int(round(v)) for v in bbox]

    # 원본 이미지 로드
    image = Image.open(img_path).convert('RGB')
    img_w, img_h = image.size

    # bbox가 이미지 범위를 벗어나지 않도록 클리핑
    # 왜 클리핑하는가: 라벨링 오류로 bbox가 이미지 경계를 넘을 수 있음
    bx = max(0, bx)
    by = max(0, by)
    bw = min(bw, img_w - bx)
    bh = min(bh, img_h - by)

    if bw <= 0 or bh <= 0:
        print(f'  [경고] bbox가 무효함: {bbox}, 건너뜀')
        return False

    # bbox 영역 crop
    cropped = image.crop((bx, by, bx + bw, by + bh))

    # 폴리곤 마스크 생성
    # 왜 마스크를 쓰는가: bbox 내부라도 상품 외부 영역(배경)을 제거해야 함
    # 폴리곤 좌표는 원본 이미지 기준이므로 crop offset만큼 빼줘야 함
    mask = Image.new('L', (bw, bh), 0)  # 흑색 배경 마스크
    draw = ImageDraw.Draw(mask)

    for polygon in segmentation:
        # 폴리곤 좌표를 (x, y) 튜플 리스트로 변환, crop offset 보정
        # COCO format: [x1, y1, x2, y2, ...] (flat list)
        poly_points = []
        for i in range(0, len(polygon), 2):
            px = polygon[i] - bx
            py = polygon[i + 1] - by
            poly_points.append((px, py))

        # 폴리곤 내부를 흰색(255)으로 채우기
        if len(poly_points) >= 3:
            draw.polygon(poly_points, fill=255)

    # 마스크를 이용하여 폴리곤 외부를 흰색으로 채우기
    # 왜 흰색인가: 상품 이미지의 배경이 일반적으로 밝색이며,
    # 모델이 배경을 무시하는 데 도움이 됨
    white_bg = Image.new('RGB', (bw, bh), (255, 255, 255))
    result = Image.composite(cropped, white_bg, mask)

    # PNG로 저장 (무손실)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path, 'PNG')
    return True


# ── 포인트 클라우드 생성 ────────────────────────────────────────────

def generate_box_point_cloud(length: float, width: float, height: float,
                             stride: float = DEFAULT_STRIDE,
                             num_points: int = DEFAULT_NUM_POINTS):
    """
    박스 치수(length, width, height)로부터 3D 박스 표면 포인트 클라우드를 생성.

    왜 박스 포인트 클라우드를 생성하는가:
      - 원본 A-Point-Set-Generation 데이터에는 3D 스캔 포인트 클라우드가 있었음
      - NIA29 샘플 데이터에는 3D 스캔이 없지만 JSON에 치수 정보가 있음
      - 상품이 차지하는 물류 공간을 박스로 근사하는 것이 "물류공간 예측" 목적에 부합
      - 박스 표면에 균일하게 점을 배치하여 모델이 3D 형태를 학습하도록 함

    좌표계:
      - 단위: cm (JSON의 치수 단위와 동일)
      - 원점: 박스 중심
      - x축: width 방향 [-width/2, width/2]
      - y축: length 방향 [-length/2, length/2]
      - z축: height 방향 [-height/2, height/2]

    점 생성 방식:
      1. 6개 면에 대해 stride 간격으로 그리드 점 생성
      2. 전체 점 수가 num_points와 다르면 무작위 복원/비복원 추출로 조정
      3. 점이 부족하면 복원 추출로 보충 (중복 허용)

    Args:
        length: 박스 길이 (cm)
        width: 박스 너비 (cm)
        height: 박스 높이 (cm)
        stride: 그리드 간격 (cm, 기본 0.5 = 5mm)
        num_points: 최종 점 수 (DataLoader 배치 처리를 위해 고정)

    Returns:
        points: (num_points, 3) numpy 배열
    """
    # 박스를 중심에 두기 위한 오프셋
    half_w = width / 2.0
    half_l = length / 2.0
    half_h = height / 2.0

    # 각 면의 그리드 점 생성
    # 왜 6개 면 모두 생성하는가: 박스의 전체 표면 형태를 모델이 학습해야 함
    all_points = []

    # x축 방향 그리드 (width 방향)
    xs = np.arange(-half_w, half_w + stride, stride)
    # y축 방향 그리드 (length 방향)
    ys = np.arange(-half_l, half_l + stride, stride)
    # z축 방향 그리드 (height 방향)
    zs = np.arange(-half_h, half_h + stride, stride)

    # 면 1, 2: z = -half_h, z = +half_h (바닥/천장)
    for z in [-half_h, half_h]:
        for x in xs:
            for y in ys:
                all_points.append([x, y, z])

    # 면 3, 4: x = -half_w, x = +half_w (좌/우)
    for x in [-half_w, half_w]:
        for y in ys:
            for z in zs:
                all_points.append([x, y, z])

    # 면 5, 6: y = -half_l, y = +half_l (전/후)
    for y in [-half_l, half_l]:
        for x in xs:
            for z in zs:
                all_points.append([x, y, z])

    all_points = np.array(all_points, dtype=np.float32)

    # 점 수를 num_points에 맞춤
    # 왜 맞추는가: DataLoader가 배치 단위로 데이터를 쌓을 때
    # 모든 샘플의 점 수가 같아야 Tensor 스택이 가능함
    current_count = len(all_points)

    if current_count == num_points:
        points = all_points
    elif current_count > num_points:
        # 점이 더 많으면 무작위 비복원 추출
        indices = np.random.choice(current_count, num_points, replace=False)
        points = all_points[indices]
    else:
        # 점이 부족하면 복원 추출로 보충
        # 왜 복원 추출인가: 점이 부족한 박스(작은 상품)도
        # 모델 입력 차원을 맞춰야 하므로 중복을 허용하여 채움
        indices = np.random.choice(current_count, num_points, replace=True)
        points = all_points[indices]

    return points


# ── JSON 라벨 저장 ─────────────────────────────────────────────────

def save_simplified_json(json_data: dict, out_path: Path):
    """
    Volume_estimation.py가 필요로 하는 필드만 포함한 간소화 JSON 저장.

    왜 간소화하는가:
      - 원본 JSON에는 segmentation, area 등 불필요한 필드가 많음
      - Volume_estimation.py는 attributes의 width/length/height만 사용
      - 파일 크기를 줄이고 디버깅을 쉽게 함

    Volume_estimation.py가 읽는 필드:
      json_data['annotations'][0]['attributes']['width']
      json_data['annotations'][0]['attributes']['length']
      json_data['annotations'][0]['attributes']['height']
    """
    ann = json_data['annotations'][0]
    attrs = ann.get('attributes', {})

    simplified = {
        'info': json_data.get('info', {}),
        'images': json_data.get('images', []),
        'annotations': [{
            'id': ann.get('id'),
            'image_id': ann.get('image_id'),
            'category_id': ann.get('category_id'),
            'bbox': ann.get('bbox'),
            'attributes': {
                'length': attrs.get('length'),
                'width': attrs.get('width'),
                'height': attrs.get('height'),
                'weight': attrs.get('weight'),
                'KAN_code': attrs.get('KAN_code'),
                'barcode': attrs.get('barcode'),
                'product_name': attrs.get('product_name'),
                'size_id': attrs.get('size_id'),
            }
        }],
        'categories': json_data.get('categories', [])
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)


# ── 데이터 분할 ────────────────────────────────────────────────────

def split_products(products, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=169):
    """
    상품(leaf 폴더) 단위로 train/val/test 분할.

    왜 상품 단위 분할인가:
      - 동일 상품의 여러 이미지가 여러 셋에 섞이면 데이터 누수 발생
      - 상품 단위 분할은 새 상품에 대한 일반화 성능을 올바르게 평가하게 함

    왜 KAN_code 기준 stratified 분할인가:
      - 단순 무작위 분할 시 샘플이 적은 클래스는 val/test에 아예 빠질 수 있음
      - stratified 분할은 각 클래스가 train/val/test에 고르게 분포하도록 보장

    왜 split_data.py의 시드(169)를 사용하는가:
      - 기존 split_data.py와 호환성 유지
      - 동일한 분할 결과를 재현할 수 있음

    Args:
        products: scan_products() 반환값
        train_ratio: 학습 비율 (기본 0.8)
        val_ratio: 검증 비율 (기본 0.1)
        test_ratio: 테스트 비율 (기본 0.1)
        seed: 랜덤 시드 (기본 169, split_data.py와 동일)

    Returns:
        (train_products, val_products, test_products) 튜플
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        'train + val + test 비율의 합이 1이어야 합니다'

    # KAN_code별로 상품 그룹화
    kan_groups = defaultdict(list)
    for p in products:
        kan_groups[p['kan_code']].append(p)

    train_products = []
    val_products = []
    test_products = []

    rng = random.Random(seed)

    for kan_code in sorted(kan_groups.keys()):
        group = kan_groups[kan_code]
        shuffled = list(group)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # 상품이 1개뿐이면 train에 배치 (val/test에 넣으면 train에 해당 클래스가 없음)
        if n == 1:
            train_products.extend(shuffled)
            continue

        # 상품이 2개면 train 1, val 1 (test 없음)
        if n == 2:
            train_products.append(shuffled[0])
            val_products.append(shuffled[1])
            continue

        train_products.extend(shuffled[:n_train])
        val_products.extend(shuffled[n_train:n_train + n_val])
        test_products.extend(shuffled[n_train + n_val:])

    return train_products, val_products, test_products


def write_split_files(train_products, val_products, test_products, dst: Path):
    """
    train/val/test 파일명 리스트를 txt 파일로 저장.

    왜 txt 파일을 생성하는가:
      - A-Point-Set-Generation의 split_data.py가 train_data.txt, val_data.txt,
        test_data.txt를 읽어 데이터 분할 정보로 사용
      - baseline_main.py가 이 txt 파일들을 읽어 DataLoader에 전달
      - 파일명은 .png 확장자를 포함해야 함 (split_data.py가 .png를 검색하기 때문)

    파일 형식:
      CSV (한 줄에 하나의 파일명)
      예: 01010102_8809162740524_1_1.png
    """
    def write_list(products, path):
        filenames = []
        for p in products:
            for img_path, _ in p['pairs']:
                # .png 확장자로 저장 (전처리 후 PNG가 되므로)
                filenames.append(img_path.stem + '.png')
        filenames.sort()

        with open(path, 'w', newline='', encoding='utf-8') as f:
            for name in filenames:
                f.write(name + '\n')

        return len(filenames)

    train_count = write_list(train_products, dst / 'train_data.txt')
    val_count = write_list(val_products, dst / 'val_data.txt')
    test_count = write_list(test_products, dst / 'test_data.txt')

    return train_count, val_count, test_count


# ── 전처리 실행 ────────────────────────────────────────────────────

def process_products(products, dst: Path, stride: float, num_points: int,
                     dry_run: bool = False):
    """
    상품 리스트에 대해 이미지 crop, 포인트 클라우드 생성, JSON 저장을 수행.

    Args:
        products: 분할된 상품 리스트
        dst: 출력 루트 경로
        stride: 포인트 클라우드 그리드 간격 (cm)
        num_points: 최종 포인트 수
        dry_run: True면 실제 파일 생성 없이 카운트만

    Returns:
        (성공 수, 실패 수, 스킵 수)
    """
    img_dst = dst / IMG_DIR
    pcd_dst = dst / PCD_DIR / PCD_SUBDIR
    json_dst = dst / JSON_DIR

    n_ok = 0
    n_fail = 0
    n_skip = 0

    for p in products:
        for img_path, json_path in p['pairs']:
            stem = img_path.stem  # 확장자 제거 (예: 01010102_8809162740524_1_1)

            # 출력 경로
            out_img = img_dst / (stem + '.png')
            out_pcd = pcd_dst / (stem + '.npy')
            out_json = json_dst / (stem + '.json')

            # 이미 모두 존재하면 스킵
            if out_img.exists() and out_pcd.exists() and out_json.exists():
                n_skip += 1
                continue

            if dry_run:
                n_ok += 1
                continue

            try:
                # JSON 로드 (UTF-8 명시 — Windows cp949 오류 방지)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                ann = json_data['annotations'][0]
                attrs = ann['attributes']

                # 치수 추출 (cm 단위)
                length = float(attrs['length'])
                width = float(attrs['width'])
                height = float(attrs['height'])

                # 치수가 0이거나 음수인 경우 건너뜀
                if length <= 0 or width <= 0 or height <= 0:
                    print(f'  [경고] 치수가 무효함 (L={length}, W={width}, H={height}): {stem}')
                    n_fail += 1
                    continue

                # 1. 이미지 crop
                if not out_img.exists():
                    success = crop_image_with_polygon(img_path, json_data, out_img)
                    if not success:
                        n_fail += 1
                        continue

                # 2. 포인트 클라우드 생성
                if not out_pcd.exists():
                    points = generate_box_point_cloud(
                        length, width, height,
                        stride=stride, num_points=num_points
                    )
                    out_pcd.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_pcd, points)

                # 3. 간소화 JSON 저장
                if not out_json.exists():
                    save_simplified_json(json_data, out_json)

                n_ok += 1

            except Exception as e:
                print(f'  [오류] {stem}: {e}')
                n_fail += 1

    return n_ok, n_fail, n_skip


# ── 통계 출력 ──────────────────────────────────────────────────────

def print_stats(products, train_products, val_products, test_products,
                n_ok, n_fail, n_skip, train_count, val_count, test_count):
    """전처리 통계 출력."""
    print()
    print('=' * 60)
    print('전처리 결과')
    print('=' * 60)
    print(f'  상품 폴더 수 — 전체: {len(products)}')
    print(f'    train: {len(train_products)}, val: {len(val_products)}, '
          f'test: {len(test_products)}')
    print(f'  이미지 처리 — 성공: {n_ok}, 실패: {n_fail}, 스킵: {n_skip}')
    print(f'  분할 파일 — train: {train_count}, val: {val_count}, test: {test_count}')
    print()

    # KAN_code 분포
    train_kans = set(p['kan_code'] for p in train_products)
    val_kans = set(p['kan_code'] for p in val_products)
    test_kans = set(p['kan_code'] for p in test_products)
    all_kans = train_kans | val_kans | test_kans

    print(f'  KAN_code 수 — 전체: {len(all_kans)}, '
          f'train: {len(train_kans)}, val: {len(val_kans)}, test: {len(test_kans)}')

    only_train = all_kans - val_kans - test_kans
    if only_train:
        print(f'  [참고] train에만 있는 KAN_code ({len(only_train)}개): '
              f'{sorted(only_train)}')
    print('=' * 60)


# ── 메인 ──────────────────────────────────────────────────────────

def clean_datasets(dst: Path):
    """출력 디렉토리 하위를 모두 삭제."""
    subdirs = [IMG_DIR, PCD_DIR, JSON_DIR, 'train_data.txt', 'val_data.txt', 'test_data.txt']
    for sd in subdirs:
        target = dst / sd
        if target.exists():
            if target.is_dir():
                print(f'  삭제: {target}/')
                shutil.rmtree(target)
            else:
                print(f'  삭제: {target}')
                target.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='NIA29 출고물품 원천데이터를 A-Point-Set-Generation 학습용으로 전처리')
    parser.add_argument('--src', default='../Sample', type=str,
                        help='원본 데이터 루트 (기본: ../Sample). '
                             '세미콜론(;)으로 구분하여 복수 경로 지정 가능. '
                             '예: --src "../Sample1;../Sample2"')
    parser.add_argument('--dst', default='datasets', type=str,
                        help='출력 데이터 루트 (기본: datasets)')
    parser.add_argument('--category', default=CATEGORY_DEFAULT, type=str,
                        help='물품 카테고리: inbound | outbound (기본: outbound)')
    parser.add_argument('--stride', default=DEFAULT_STRIDE, type=float,
                        help='포인트 클라우드 그리드 간격 cm (기본: 0.5 = 5mm)')
    parser.add_argument('--num_points', default=DEFAULT_NUM_POINTS, type=int,
                        help='포인트 클라우드 점 수 (기본: 11000, 모델 num_points와 일치)')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='학습셋 비율 (기본: 0.8)')
    parser.add_argument('--val_ratio', default=0.1, type=float,
                        help='검증셋 비율 (기본: 0.1)')
    parser.add_argument('--test_ratio', default=0.1, type=float,
                        help='테스트셋 비율 (기본: 0.1)')
    parser.add_argument('--seed', default=169, type=int,
                        help='분할 랜덤 시드 (기본: 169, split_data.py와 동일)')
    parser.add_argument('--clean', action='store_true',
                        help='기존 출력 디렉토리 삭제 후 재생성')
    parser.add_argument('--dry_run', action='store_true',
                        help='파일 생성 없이 통계만 출력')
    args = parser.parse_args()

    dst = Path(args.dst).resolve()

    # --src 인자를 세미콜론으로 분리하여 복수 소스 경로 리스트로 변환.
    # 왜 세미콜론인가: Windows 경로에 콜론(:)이 드라이브 문자에 사용되므로
    # 콜론을 구분자로 쓸 수 없고, 세미콜론은 Windows 환경 변수 PATH 구분자이기도 해
    # 사용자에게 익숙한 방식이다.
    src_paths = [s.strip() for s in args.src.split(';') if s.strip()]
    src_paths = [Path(s).resolve() for s in src_paths]

    category_folder = CATEGORY_MAP.get(args.category, args.category)

    print(f'원본 소스: {len(src_paths)}개')
    for i, src in enumerate(src_paths):
        print(f'  [{i+1}] {src}')
    print(f'출력: {dst}')
    print(f'카테고리: {args.category} → {category_folder}')
    print(f'포인트 클라우드: stride={args.stride}cm, num_points={args.num_points}')
    print(f'분할 비율: train={args.train_ratio}, val={args.val_ratio}, '
          f'test={args.test_ratio} (시드: {args.seed})')
    print(f'모드: {"dry-run" if args.dry_run else "실제 전처리"}')
    print()

    # 1. 스캔 — 모든 소스 경로에서 상품을 수집하여 하나의 리스트로 합친다.
    # 왜 합쳐서 스캔하는가: 분할(split) 시 모든 소스의 상품을 함께 고려해야
    # KAN_code 기준 stratified 분할이 올바르게 작동하기 때문이다.
    print('[1/5] 원본 데이터 스캔 중...')
    products = []
    for i, src in enumerate(src_paths):
        print(f'  소스 [{i+1}/{len(src_paths)}] 스캔: {src}')
        src_products = scan_products(src, category=args.category)
        print(f'    상품 폴더: {len(src_products)}개, '
              f'이미지-JSON 쌍: {sum(len(p["pairs"]) for p in src_products)}개')

        # 중복 상품 폴더명 검사 — 다른 소스에 같은 product_folder가 있으면 경고.
        # 왜 검사하는가: 파일명이 동일하면 출력 경로가 충돌하여 한쪽이 덮어씌워지기 때문.
        existing_folders = {p['product_folder'] for p in products}
        for p in src_products:
            if p['product_folder'] in existing_folders:
                print(f'    [경고] 중복 상품 폴더: {p["product_folder"]} '
                      f'(이전 소스에도 존재함, 파일이 덮어씌워질 수 있음)')
        products.extend(src_products)

    print(f'  합계 — 상품 폴더: {len(products)}개')
    total_pairs = sum(len(p['pairs']) for p in products)
    print(f'  이미지-JSON 쌍: {total_pairs}개')

    kan_counts = defaultdict(int)
    for p in products:
        kan_counts[p['kan_code']] += 1
    print(f'  KAN_code 종류: {len(kan_counts)}개')

    # 2. 분할
    print()
    print('[2/5] train / val / test 분할 중...')
    train_products, val_products, test_products = split_products(
        products,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print(f'  train: {len(train_products)}개 상품, '
          f'val: {len(val_products)}개 상품, test: {len(test_products)}개 상품')

    # 3. 클린 (옵션)
    if args.clean and not args.dry_run:
        print()
        print('[3/5] 기존 출력 삭제 중...')
        clean_datasets(dst)

    # 4. 전처리
    print()
    print('[4/5] 전처리 중... (이미지 crop + 포인트 클라우드 생성 + JSON 저장)')
    all_products = train_products + val_products + test_products
    n_ok, n_fail, n_skip = process_products(
        all_products, dst,
        stride=args.stride, num_points=args.num_points,
        dry_run=args.dry_run
    )
    print(f'  성공: {n_ok}, 실패: {n_fail}, 스킵(이미 존재): {n_skip}')

    # 5. 분할 파일 저장
    print()
    print('[5/5] 분할 파일 저장 중...')
    if not args.dry_run:
        train_count, val_count, test_count = write_split_files(
            train_products, val_products, test_products, dst
        )
        print(f'  train_data.txt: {train_count}개')
        print(f'  val_data.txt: {val_count}개')
        print(f'  test_data.txt: {test_count}개')
    else:
        train_count = sum(len(p['pairs']) for p in train_products)
        val_count = sum(len(p['pairs']) for p in val_products)
        test_count = sum(len(p['pairs']) for p in test_products)
        print(f'  (dry-run) train: {train_count}, val: {val_count}, test: {test_count}')

    # 통계
    print_stats(products, train_products, val_products, test_products,
                n_ok, n_fail, n_skip, train_count, val_count, test_count)

    # 다음 단계 안내
    print()
    print('출력 디렉토리 구조:')
    print(f'  {dst}/')
    print(f'    {IMG_DIR}/              ← 폴리곤 crop PNG 이미지')
    print(f'    {PCD_DIR}/{PCD_SUBDIR}/     ← 박스 포인트 클라우드 NPY')
    print(f'    {JSON_DIR}/              ← 치수 JSON')
    print(f'    train_data.txt           ← 학습 파일명 리스트')
    print(f'    val_data.txt             ← 검증 파일명 리스트')
    print(f'    test_data.txt            ← 테스트 파일명 리스트')
    print()
    print('다음 단계 — baseline_main.py의 경로를 수정하고 학습:')
    print(f'  image_root = r"{dst / IMG_DIR}"')
    print(f'  point_cloud_root = r"{dst / PCD_DIR}"')
    print(f'  python baseline_main.py --training True')
    print()
    print('  ※ Windows에서는 num_workers=0, batch_size=4~8 권장')


if __name__ == '__main__':
    main()
