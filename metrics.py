import torch


# =============================================================================
# [모듈 개요]
# 이 파일은 A-Point-Set-Generation 프로젝트의 3D 평가 지표 계산 모듈이다.
# 단일 이미지로부터 생성된 3D 포인트 클라우드(예측값)와 정답(GT) 포인트
# 클라우드 간의 형태 일치도를 평가하기 위해 3D IoU(Intersection over Union)
# 와 mIoU(mean IoU)를 계산한다.
#
# IoU는 두 3D 영역(바운딩 박스)이 얼마나 겹치는지를 0~1 사이 값으로 나타내며,
# 1에 가까울수록 예측이 정답과 형태적으로 유사함을 의미한다.
#
# 본 파일에는 두 가지 버전의 IoU 계산 방식이 존재한다:
#   (1) 정규화 기반 IoU  ← 현재 활성 코드 (GT 기준으로 [0,1] 정규화 후 비교)
#   (2) 복셀(voxel) 기반 IoU  ← 문자열 주석로 비활성화된 대체 구현
# 정규화 기반 방식이 빠르고 단순하여 활성 코드로 채택되었다.
# =============================================================================


def normalize_points(points, reference_points=None):
    """포인트 클라우드를 0~1 범위로 정규화하여 스케일/위치 불변 비교를 가능하게 한다."""
    # -------------------------------------------------------------------------
    # [왜 정규화가 필요한가?]
    # 포인트 클라우드의 절대 좌표는 물체의 크기나 원점 위치에 따라 크게 달라진다.
    # 예를 들어 같은 박스라도 원점에서 멀리 떨어져 있으면 좌표값이 커지고,
    # 작게 스캔되면 좌표값이 작아진다. 이 상태로는 서로 다른 샘플 간의
    # 형태 일치도를 공정하게 비교할 수 없다.
    # 따라서 GT(정답)의 바운딩 박스를 기준으로 좌표를 [0, 1] 구간으로 옮기면
    # 크기와 위치의 영향을 제거하고 "형태" 자체만 비교할 수 있다.
    # -------------------------------------------------------------------------
    # Args:
    #     points: (N, 3) tensor - 정규화할 포인트 클라우드
    #     reference_points: (N, 3) tensor - 기준이 되는 포인트 클라우드 (보통 ground truth)
    # Returns:
    #     normalized_points: (N, 3) tensor - 정규화된 포인트 클라우드
    # -------------------------------------------------------------------------
    if reference_points is None:
        # 별도의 기준이 주어지지 않았다면 자기 자신을 기준으로 삼는다.
        # 이 경우 points 자체의 바운딩 박스가 [0,1]이 되므로 독립 정규화가 된다.
        reference_points = points

    # 기준 포인트 클라우드의 바운딩 박스(각 축의 최소/최대 좌표)를 구한다.
    # 이 바운딩 박스가 정규화의 "자" 역할을 한다.
    ref_min = reference_points.min(dim=0)[0]
    ref_max = reference_points.max(dim=0)[0]
    ref_scale = ref_max - ref_min

    # (points - ref_min): 원점을 ref_min으로 평행이동하여 최소점이 0이 되게 한다.
    # /(ref_scale + 1e-8): 바운딩 박스 크기로 나누어 [0,1] 범위로 압축한다.
    # 1e-8을 더하는 이유는 ref_scale이 0(모든 점이 한 점에 모인 퇴화 케이스)일 때
    # 0으로 나누는 오류(division by zero)를 방지하기 위해서다.
    normalized_points = (points - ref_min) / (ref_scale + 1e-8)

    return normalized_points


def calculate_3d_bbox(points):
    """포인트 클라우드를 감싸는 최소 3D 바운딩 박스(축 정렬)의 min/max 좌표를 계산한다."""
    # -------------------------------------------------------------------------
    # [왜 바운딩 박스를 구하는가?]
    # 포인트 클라우드는 수만 개의 점으로 이루어져 있어 직접 비교하기 어렵다.
    # 대신 점들을 감싸는 가장 작은 직육면체(바운딩 박스)를 구하면,
    # 두 물체의 위치·크기·형태를 단순한 6개 좌표(min 3 + max 3)로 요약할 수 있다.
    # 축 정렬(Axis-Aligned) 박스를 사용하는 이유는 계산이 단순하고 빠르기 때문이다.
    # -------------------------------------------------------------------------
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    return min_coords, max_coords


def calculate_volume(min_coords, max_coords):
    """Bounding Box 부피 계산"""
    # -------------------------------------------------------------------------
    # [왜 부피가 필요한가?]
    # IoU = 교집합 / 합집합 이다.
    # 합집합 부피 = pred 부피 + gt 부피 - 교집합 부피 로 계산되므로
    # 각 박스의 부피가 먼저 필요하다. 부피는 x,y,z 길이의 곱이다.
    # -------------------------------------------------------------------------
    # Args:
    #     min_coords: (3,) tensor [x_min, y_min, z_min]
    #     max_coords: (3,) tensor [x_max, y_max, z_max]
    # Returns:
    #     volume: 바운딩 박스의 부피
    # -------------------------------------------------------------------------
    lengths = max_coords - min_coords  # x, y, z 길이
    volume = lengths.prod()  # 부피 = x * y * z
    return volume


def calculate_intersection_volume(min1, max1, min2, max2):
    """두 박스의 교차 부피 계산"""
    # -------------------------------------------------------------------------
    # [왜 이렇게 교집합을 구하는가?]
    # 두 축 정렬 박스의 교집합은 각 축마다 "더 큰 최소값"과 "더 작은 최대값" 사이의
    # 구간이 된다. 어떤 축이라도 겹치지 않으면 교집합은 0이 되어야 한다.
    # 따라서 (max - min)이 음수가 되면 0으로 잘라주는 clamp가 필수적이다.
    # -------------------------------------------------------------------------
    # Args:
    #     min1, max1: 첫 번째 박스의 최소/최대 좌표
    #     min2, max2: 두 번째 박스의 최소/최대 좌표
    # Returns:
    #     intersection_volume: 교차 부피
    # -------------------------------------------------------------------------
    intersection_min = torch.max(min1, min2)  # 교차 영역의 최소 좌표
    intersection_max = torch.min(max1, max2)  # 교차 영역의 최대 좌표

    # 교차 길이가 음수인 경우, 겹침이 없음
    # clamp(min=0): 겹치지 않는 축이 하나라도 있으면 그 축 길이를 0으로 만들어
    # 전체 부피도 0이 되도록 한다(빈 교집합 처리).
    intersection_lengths = torch.clamp(intersection_max - intersection_min, min=0)
    intersection_volume = intersection_lengths.prod()  # 부피 = x * y * z
    return intersection_volume


def calculate_3d_iou(pred_points, gt_points):
    """정규화된 공간에서 3D IoU 계산"""
    # -------------------------------------------------------------------------
    # [왜 GT 기준으로 정규화한 뒤 IoU를 구하는가?]
    # 예측과 정답을 동일한 "자(GT 바운딩 박스)"로 정규화하면,
    # GT는 정확히 [0,1] 박스가 되고 예측은 GT에 상대적인 박스가 된다.
    # 이렇게 하면 스케일/위치 차이가 제거되어 형태 일치도만 남게 되므로,
    # 서로 다른 크기의 물체에 대해서도 공정한 IoU 비교가 가능하다.
    # -------------------------------------------------------------------------
    # Args:
    #     pred_points: (B, N, 3) tensor of predicted point coordinates
    #     gt_points: (B, N, 3) tensor of ground truth point coordinates
    # Returns:
    #     iou_scores: (B,) tensor of IoU scores
    # -------------------------------------------------------------------------
    batch_size = pred_points.shape[0]
    ious = []

    for i in range(batch_size):
        # Ground truth를 기준으로 두 포인트 클라우드 정규화
        # GT는 자기 자신을 기준으로 정규화 → [0,1] 박스
        # pred는 GT의 min/scale을 빼고 나눔 → GT 좌표계에서의 상대 박스
        normalized_gt = normalize_points(gt_points[i])
        normalized_pred = normalize_points(pred_points[i], gt_points[i])

        # 정규화된 공간에서 바운딩 박스 계산
        pred_min, pred_max = calculate_3d_bbox(normalized_pred)
        gt_min, gt_max = calculate_3d_bbox(normalized_gt)

        # 정규화된 공간에서 각 박스의 부피 계산
        pred_volume = calculate_volume(pred_min, pred_max)
        gt_volume = calculate_volume(gt_min, gt_max)

        # 정규화된 공간에서 교차 영역 부피 계산
        intersection_volume = calculate_intersection_volume(pred_min, pred_max, gt_min, gt_max)

        # Union Volume 계산
        # 합집합 = A + B - 교집합 (포함-배제 원리)으로 중복 계산을 제거한다.
        union_volume = pred_volume + gt_volume - intersection_volume

        # IoU 계산
        # 1e-8을 더해 0으로 나누는 것을 방지한다(두 박스가 모두 퇴화한 경우).
        iou = intersection_volume / (union_volume + 1e-8)  # 0으로 나누는 것 방지
        ious.append(iou)

    return torch.tensor(ious, device=pred_points.device)


def calculate_3d_miou(pred_points, gt_points, num_classes=1):
    """정규화된 공간에서 평균 3D IoU 계산"""
    # -------------------------------------------------------------------------
    # [왜 평균(mIoU)을 구하는가?]
    # 배치 내 각 샘플마다 IoU가 다르므로, 전체 성능을 하나의 숫자로 요약하려면
    # 평균이 필요하다. mIoU는 모델의 전반적인 형태 재구성 품질을 대표한다.
    # num_classes는 다중 클래스 확장을 위해 남겨둔 파라미터이나,
    # 본 프로젝트는 단일 클래스(박스)이므로 현재는 사용하지 않는다.
    # -------------------------------------------------------------------------
    # Args:
    #     pred_points: (B, N, 3) tensor of predicted points
    #     gt_points: (B, N, 3) tensor of ground truth points
    #     num_classes: Number of object classes (현재는 사용하지 않음)
    # Returns:
    #     Mean IoU across batch
    # -------------------------------------------------------------------------
    iou_scores = calculate_3d_iou(pred_points, gt_points)
    mean_iou = iou_scores.mean()

    return mean_iou


def calculate_3d_iou_with_physical_scale(pred_points, gt_points, physical_scale):
    """실제 물리적 크기를 고려한 3D IoU 계산 (optional)"""
    # -------------------------------------------------------------------------
    # [왜 물리적 스케일로 나누는가?]
    # 정규화 기반 IoU는 형태만 비교하므로 실제 크기 차이를 반영하지 못한다.
    # 물리적 스케일(가로/세로/높이)로 좌표를 나누면, 단위가 동일한 물리 공간에서
    # 비교하므로 실제 크기까지 고려한 IoU를 얻을 수 있다.
    # 이후 계산은 정규화 버전의 calculate_3d_iou를 그대로 재사용한다.
    # -------------------------------------------------------------------------
    # Args:
    #     pred_points: (B, N, 3) tensor of predicted points
    #     gt_points: (B, N, 3) tensor of ground truth points
    #     physical_scale: (3,) tensor or list - 실제 물리적 크기 [width, height, depth]
    # Returns:
    #     Mean IoU across batch
    # -------------------------------------------------------------------------
    physical_scale = torch.tensor(physical_scale, device=pred_points.device)
    normalized_pred = pred_points / physical_scale
    normalized_gt = gt_points / physical_scale
    return calculate_3d_iou(normalized_pred, normalized_gt)


# =============================================================================
# [비활성 대체 구현 #1 - 정규화 없이 원본 좌표계에서 직접 IoU 계산]
# 아래 블록은 문자열 주석(""" """)로 감싸져 있어 실행되지 않는다.
# 정규화 단계를 생략하고 원래 좌표계에서 바로 바운딩 박스를 구하는 버전이다.
# 스케일/위치가 서로 다른 샘플 간 비교에 불리하여 정규화 버전으로 대체되었다.
# 원본 코드 보존을 위해 그대로 유지한다.
# =============================================================================
"""
import torch

def calculate_3d_bbox(points):
    min_coords = points.min(dim=0)[0]  # 각 축의 최소값
    max_coords = points.max(dim=0)[0]  # 각 축의 최대값
    return min_coords, max_coords

def calculate_volume(min_coords, max_coords):
    lengths = max_coords - min_coords  # x, y, z 길이
    volume = lengths.prod()  # 부피 = x * y * z
    return volume

def calculate_intersection_volume(min1, max1, min2, max2):
    intersection_min = torch.max(min1, min2)  # 교차 영역의 최소 좌표
    intersection_max = torch.min(max1, max2)  # 교차 영역의 최대 좌표
    
    # 교차 길이가 음수인 경우, 겹침이 없음
    intersection_lengths = torch.clamp(intersection_max - intersection_min, min=0)
    intersection_volume = intersection_lengths.prod()  # 부피 = x * y * z
    return intersection_volume

def calculate_3d_iou(pred_points, gt_points):
    batch_size = pred_points.shape[0]
    ious = []
    
    for i in range(batch_size):
        # 예측 및 실제 포인트 클라우드의 Bounding Box 계산
        pred_min, pred_max = calculate_3d_bbox(pred_points[i])
        gt_min, gt_max = calculate_3d_bbox(gt_points[i])
        
        # 각 박스의 부피 계산
        pred_volume = calculate_volume(pred_min, pred_max)
        gt_volume = calculate_volume(gt_min, gt_max)
        
        # 교차 영역 부피 계산
        intersection_volume = calculate_intersection_volume(pred_min, pred_max, gt_min, gt_max)
        
        # Union Volume 계산
        union_volume = pred_volume + gt_volume - intersection_volume
        
        # IoU 계산
        iou = intersection_volume / (union_volume + 1e-8)  # 0으로 나누는 것 방지
        ious.append(iou)
    
    return torch.tensor(ious, device=pred_points.device)

def calculate_3d_miou(pred_points, gt_points, num_classes=1):

    iou_scores = calculate_3d_iou(pred_points, gt_points)
    mean_iou = iou_scores.mean()
    
    return mean_iou
"""

# =============================================================================
# [비활성 대체 구현 #2 - 복셀(voxel) 기반 IoU 계산]
# 아래 블록도 문자열 주석로 감싸져 있어 실행되지 않는다.
# 포인트 클라우드를 일정 크기의 3D 격자(복셀)로 양자화한 뒤,
# 겹치는 복셀 수로 IoU를 구하는 방식이다.
# 바운딩 박스 기반보다 형태를 더 정밀하게 반영하지만,
# 복셀 해상도에 따라 결과가 달라지고 계산 비용이 크기 때문에
# 정규화 기반 바운딩 박스 버전이 활성 코드로 채택되었다.
# 원본 코드 보존을 위해 그대로 유지한다.
# =============================================================================
"""
def voxelize_pointcloud(points, voxel_size=8):# 8, 16, 32, 64, 128
    # Normalize points to [0, voxel_size-1]
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    points = (points - min_coords) * (voxel_size-1) / (max_coords - min_coords)
    
    # Convert to voxel indices
    indices = points.long()
    indices = torch.clamp(indices, 0, voxel_size-1)
    
    # Create voxel grid
    voxels = torch.zeros((voxel_size, voxel_size, voxel_size), device=points.device)
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    
    return voxels

def calculate_3d_iou(pred_points, gt_points, voxel_size=8):
    batch_size = pred_points.shape[0]
    ious = []
    
    for i in range(batch_size):
        # Convert pointclouds to voxel representation
        pred_voxels = voxelize_pointcloud(pred_points[i], voxel_size)
        gt_voxels = voxelize_pointcloud(gt_points[i], voxel_size)
        
        # Calculate intersection and union
        intersection = torch.logical_and(pred_voxels, gt_voxels).sum().float()
        union = torch.logical_or(pred_voxels, gt_voxels).sum().float()
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        ious.append(iou)
    
    return torch.tensor(ious, device=pred_points.device)

def calculate_3d_miou(pred_points, gt_points, num_classes=1, voxel_size=8):

    total_iou = 0.0
    
    # For now we only have one class, but this can be extended
    iou_scores = calculate_3d_iou(pred_points, gt_points, voxel_size)
    mean_iou = iou_scores.mean()
    
    return mean_iou
"""
