# =============================================================================
# split_data.py — 데이터 분할 유틸리티 (원본)
# -----------------------------------------------------------------------------
# 본 스크립트는 전체 이미지 데이터를 train / val / test 세 가지로 무작위 분할하여
# 각각의 파일명 리스트를 텍스트 파일로 저장한다.
#
# ※ 주의: 이 스크립트는 원본 유틸리티이며, 파일명 단위로만 분할을 수행한다.
#    상품(product) 단위로 분할하여 데이터 누수(data leakage)를 방지하는
#    prepare_data_3d.py 의 분할 로직이 더 향상되어 있으므로 현재는
#    prepare_data_3d.py 사용을 권장한다.
#
# ※ baseline_main.py 는 train/val/test txt 파일이 이미 존재하면
#    split_data()를 호출하지 않고 바로 학습으로 넘어가므로,
#    이 함수가 실제로 실행되는 경우는 드물다.
# =============================================================================

# os: 파일/디렉토리 존재 여부 확인 및 경로 순회를 위해 사용
# csv: 파일명 리스트를 CSV 형식으로 읽고 쓰기 위해 사용
#      (단순 텍스트도 가능하지만, CSV 형식이 한 줄에 하나의 파일명을
#       안전하게 저장·파싱할 수 있어 호환성이 좋다)
# random: 데이터를 무작위로 섞어 train/val/test를 분할하기 위해 사용
import os, csv, random


def read_from_file(path):
    """
    파일명 리스트가 저장된 CSV 텍스트 파일을 읽어와서 리스트로 반환한다.

    학습/검증/테스트용 파일명 리스트가 텍스트 파일로 저장되어 있으므로,
    이를 다시 메모리로 불러와 데이터 로더에 전달하기 위해 사용한다.
    """
    data = []

    # newline='' 없이 열면 일부 플랫폼에서 빈 줄이 추가될 수 있으나,
    # 읽기 모드에서는 큰 문제가 없어 단순히 'r' 모드로 연다.
    with open(path, 'r') as file:
        # csv.reader를 사용하면 쉼표로 구분된 필드를 안전하게 분리할 수 있다.
        # 파일명에 쉼표가 포함될 가능성에 대비한 견고한 파싱을 위해 CSV 형식을 유지한다.
        reader = csv.reader(file)
        for line in reader:
            if line:  # 빈 라인 건너뛰기 — 파일 끝에 빈 줄이 있을 수 있어 방어적으로 검사
                # 각 행의 첫 번째 열(파일명)만 사용한다.
                # write_to_file에서 파일명을 단일 열로 저장하므로 line[0]가 파일명이다.
                data.append(line[0])

    return data


def write_to_file(path, data):
    """
    파일명 리스트를 CSV 텍스트 파일로 저장한다.

    분할된 train/val/test 파일명 리스트를 디스크에 저장해 두면,
    이후 학습 스크립트가 매번 분할을 다시 수행할 필요 없이
    동일한 분할 결과를 재현할 수 있다.
    """
    # newline=''을 지정해야 Windows 환경에서 csv.writer가
    # 줄바꿈을 두 번写入하는 현상(빈 줄 삽입)을 방지할 수 있다.
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for filename in data:
            # 파일명을 단일 열(리스트)로 감싸서 한 줄에 하나씩 기록한다.
            # 이렇게 하면 read_from_file에서 line[0]로 쉽게 읽어올 수 있다.
            writer.writerow([filename])


def split_data(train_ratio, val_ratio, test_ratio, overrideFiles=False):
    """
    이미지 디렉토리를 순회하여 모든 PNG 파일명을 수집한 뒤,
    지정된 비율(train/val/test)에 따라 무작위 분할하여 각각 텍스트 파일로 저장한다.

    ※ 주의: 이 함수 내부에 하드코딩된 경로(/workspace/DATA/...)가 존재한다.
       하지만 baseline_main.py에서 이미 train/val/test txt 파일이 있으면
       이 함수는 실행되지 않으므로 실제 영향은 제한적이다.

    Args:
        train_ratio (float): 학습 데이터 비율 (예: 0.8)
        val_ratio   (float): 검증 데이터 비율 (예: 0.1)
        test_ratio  (float): 테스트 데이터 비율 (예: 0.1)
        overrideFiles (bool): True이면 기존 분할 파일이 있어도 덮어쓴다.
                              False이면 기존 파일이 모두 존재할 때 분할을 건너뛴다.
    """

    # --- 1. 기존 분할 파일 존재 여부 확인 ---
    # 학습/검증/테스트 분할 결과가 이미 파일로 존재하면,
    # 매번 다시 분할할 필요가 없다 (동일한 분할을 재현하기 위함).
    path_train = 'train_data.txt'
    path_val   = 'val_data.txt'
    path_test  = 'test_data.txt'

    # 세 파일이 모두 존재하고 overrideFiles가 False면,
    # 기존 분할 결과를 재사용하도록 함수를 종료한다.
    # overrideFiles=True일 때만 강제로 다시 분할한다.
    if (os.path.isfile(path_train)
            and os.path.isfile(path_val)
            and os.path.isfile(path_test)
            and overrideFiles == False):
        return

    # --- 2. 데이터 경로 설정 (하드코딩) ---
    # 원본 서버 환경의 경로가 하드코딩되어 있다.
    # prepare_data_3d.py가 생성한 datasets/ 경로와 다를 수 있으므로 주의.
    image_root       = "/workspace/DATA/NIA29_3D/images_poly_bbox_crop"
    point_cloud_root = "/workspace/DATA/NIA29_3D/labels"

    # 수집된 파일명을 저장할 리스트
    data = []

    # --- 3. 모든 PNG 파일명 수집 ---
    # os.walk로 하위 디렉토리까지 재귀적으로 순회하는 이유는
    # 이미지가 카테고리별 하위 폴더에 나뉘어 저장되어 있을 수 있기 때문이다.
    for root, dirs, files in os.walk(image_root):
        for file in files:
            if file.endswith('.png'):  # PNG 파일만 처리 — 라벨 이미지 등 비-PNG 파일을 제외하기 위해
                data.append(file)      # 파일명만 저장 (경로 제외) — 데이터 로더가 루트 경로 기준으로 찾는다

    # --- 4. 재현 가능한 무작위 섞기 ---
    # random.seed를 고정(169)하는 이유:
    # 매번 실행할 때마다 동일한 train/val/test 분할 결과를 얻기 위해서다.
    # 시드를 고정하지 않으면 실행마다 분할이 달라져
    # 학습 결과를 비교하거나 버그를 재현하기 어려워진다.
    random.seed(169)
    random.shuffle(data)

    # --- 5. 비율에 따라 train / val / test 분할 ---
    # 슬라이싱을 이용해 리스트를 세 구간으로 나눈다.
    # 비율의 합이 1.0이 되도록 호출자가 보장해야 한다.

    # 앞부분 train_ratio 만큼을 학습 데이터로 사용
    train_data = data[:int(train_ratio * len(data))]

    # train 다음 구간부터 (train+val) 구간까지를 검증 데이터로 사용
    val_data = data[int(train_ratio * len(data)):
                    int((train_ratio + val_ratio) * len(data))]

    # (train+val) 구간 이후 끝까지를 테스트 데이터로 사용
    test_data = data[int((train_ratio + val_ratio) * len(data)):
                     int((train_ratio + val_ratio + test_ratio) * len(data))]

    # --- 6. 분할 결과를 파일로 저장 ---
    # write_to_file 함수를 이용해 각 분할 결과를 텍스트 파일로 저장한다.
    # 이후 학습/검증/테스트 스크립트는 이 파일들을 읽어 데이터를 로드한다.
    write_to_file(path_train, train_data)
    write_to_file(path_val,   val_data)
    write_to_file(path_test,  test_data)
