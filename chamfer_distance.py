"""
chamfer_distance.py
===================
순수 PyTorch 구현 Chamfer Distance 모듈.

원본 pytorch3d의 `chamfer_distance` 모듈과 동일한 API를 제공하기 위해 작성됨.
Windows에서 pytorch3d를 소스 빌드하려면 CUDA Toolkit + MSVC가 필요하므로,
대신 이 순수 PyTorch 구현을 사용하여 컴파일 없이 GPU 가속 지원.

API (기존 코드와 호환):
    from chamfer_distance import ChamferDistance
    chamferDist = ChamferDistance()
    dist1, dist2 = chamferDist(pred, point_cloud)
    # 또는
    dist = chamferDist(pred, point_cloud)
    # dist[0] == dist1, dist[1] == dist2

    dist1: [B, N] — pred의 각 점에서 point_cloud의 최근접 점까지의 거리
    dist2: [B, M] — point_cloud의 각 점에서 pred의 최근접 점까지의 거리

메모리 최적화:
    11000×11000 거리 행렬을 한 번에 계산하면 배치 크기에 따라
    수 GB의 메모리가 필요하므로, 청크 단위로 분할하여 계산.
"""

import torch
import torch.nn as nn


class ChamferDistance(nn.Module):
    """
    Chamfer Distance를 순수 PyTorch로 구현한 클래스.

    원본 pytorch3d의 chamfer_distance.ChamferDistance와 동일한 인터페이스:
        chamferDist = ChamferDistance()
        dist1, dist2 = chamferDist(x, y)

    x: [B, N, 3] 예측 포인트 클라우드
    y: [B, M, 3] 정답 포인트 클라우드

    반환값:
        dist1: [B, N] — x의 각 점에서 y의 최근접 점까지의 제곱 거리
        dist2: [B, M] — y의 각 점에서 x의 최근접 점까지의 제곱 거리

    왜 제곱 거리인가:
        원본 pytorch3d 구현이 제곱 거리(squared distance)를 반환하므로
        동일한 동작을 보장하기 위해 제곱 거리를 사용.
    """

    def forward(self, x, y):
        """
        Args:
            x: [B, N, D] 첫 번째 포인트 클라우드
            y: [B, M, D] 두 번째 포인트 클라우드

        Returns:
            (dist1, dist2) 튜플
            dist1: [B, N] — x→y 최근접 거리 (제곱)
            dist2: [B, M] — y→x 최근접 거리 (제곱)
        """
        return _chamfer_distance(x, y)


def _chamfer_distance(x, y, chunk_size=2048):
    """
    청크 단위로 Chamfer Distance를 계산.

    왜 청크 단위인가:
        torch.cdist(x, y)는 [B, N, M] 크기의 거리 행렬을 생성.
        N=M=11000, B=4인 경우 4×11000×11000×4bytes ≈ 1.9GB 필요.
        8GB GPU에서는 메모리 부족 가능성이 있으므로 청크로 분할.

    Args:
        x: [B, N, D]
        y: [B, M, D]
        chunk_size: 한 번에 처리할 x의 점 수

    Returns:
        (dist1, dist2) 튜플
    """
    batch_size, n_points, _ = x.shape
    _, m_points, _ = y.shape

    # 배치 크기에 따라 청크 크기를 자동 조정한다.
    # 왜 자동 조정인가: chunk_size=2048, batch_size=32, M=11000인 경우
    # 거리 행렬 하나만에 32×2048×11000×4bytes ≈ 2.9GB가 필요하여
    # 8GB GPU에서 메모리 부족(OOM)이 발생하기 때문이다.
    # 목표: 단일 거리 행렬이 약 512MB 이하가 되도록 청크 크기를 줄인다.
    # 512MB = 512*1024*1024 bytes, float32(4bytes) 기준
    # max_chunk = 512MB / (batch_size * m_points * 4bytes)
    max_chunk_bytes = 512 * 1024 * 1024
    max_chunk = max(64, int(max_chunk_bytes / (batch_size * m_points * 4)))
    if chunk_size > max_chunk:
        chunk_size = max_chunk

    # 점 수가 적으면 한 번에 계산 (메모리 충분)
    if n_points * m_points <= chunk_size * chunk_size:
        return _chamfer_distance_full(x, y)

    # 청크 단위로 dist1 (x→y) 계산
    dist1_list = []
    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        x_chunk = x[:, i:end, :]  # [B, chunk, D]
        # 제곱 거리 행렬: [B, chunk, M]
        dist_matrix = _squared_distance_matrix(x_chunk, y)
        dist1_list.append(dist_matrix.min(dim=2)[0])  # [B, chunk]

    dist1 = torch.cat(dist1_list, dim=1)  # [B, N]

    # 청크 단위로 dist2 (y→x) 계산
    dist2_list = []
    for j in range(0, m_points, chunk_size):
        end = min(j + chunk_size, m_points)
        y_chunk = y[:, j:end, :]  # [B, chunk, D]
        # 제곱 거리 행렬: [B, chunk, N]
        dist_matrix = _squared_distance_matrix(y_chunk, x)
        dist2_list.append(dist_matrix.min(dim=2)[0])  # [B, chunk]

    dist2 = torch.cat(dist2_list, dim=1)  # [B, M]

    return dist1, dist2


def _chamfer_distance_full(x, y):
    """메모리가 충분할 때 한 번에 전체 거리 행렬을 계산."""
    dist_matrix = _squared_distance_matrix(x, y)  # [B, N, M]
    dist1 = dist_matrix.min(dim=2)[0]  # [B, N]
    dist2 = dist_matrix.min(dim=1)[0]  # [B, M]
    return dist1, dist2


def _squared_distance_matrix(x, y):
    """
    두 포인트 클라우드 간의 제곱 거리 행렬을 계산.

    ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y

    왜 이 공식을 쓰는가:
        torch.cdist보다 메모리 효율적이고 자동미분이 안정적.
        음수 방지를 위해 clamp 적용.

    Args:
        x: [B, N, D]
        y: [B, M, D]

    Returns:
        [B, N, M] 제곱 거리 행렬
    """
    # ||x||^2: [B, N, 1]
    x_norm = (x ** 2).sum(dim=2, keepdim=True)
    # ||y||^2: [B, 1, M]
    y_norm = (y ** 2).sum(dim=2, keepdim=True).transpose(1, 2)
    # x·y: [B, N, M]
    dot = torch.bmm(x, y.transpose(1, 2))

    # 제곱 거리 (음수 방지)
    dist = x_norm + y_norm - 2.0 * dot
    dist = torch.clamp(dist, min=0.0)

    return dist
