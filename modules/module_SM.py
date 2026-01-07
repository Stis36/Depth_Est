"""
Stereo Matching functions: ステレオマッチング関連の関数
PatchMatch + NCC による視差推定を実装
"""

import cv2
import numpy as np
from typing import Tuple


def stereo_rectify(img1: np.ndarray, img2: np.ndarray,
                   camera_matrix1: np.ndarray, dist_coeffs1: np.ndarray,
                   camera_matrix2: np.ndarray, dist_coeffs2: np.ndarray,
                   R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ステレオ校正を実行
    
    Args:
        img1, img2: 入力画像
        camera_matrix1, dist_coeffs1: カメラ1の内部パラメータ
        camera_matrix2, dist_coeffs2: カメラ2の内部パラメータ
        R, T: カメラ間の回転行列と並進ベクトル
    
    Returns:
        rectified_img1, rectified_img2: 校正された画像
        Q: 再投影行列
    """
    img_size = (img1.shape[1], img1.shape[0])
    
    # ステレオ校正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1=camera_matrix1,
        distCoeffs1=dist_coeffs1,
        cameraMatrix2=camera_matrix2,
        distCoeffs2=dist_coeffs2,
        imageSize=img_size,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9
    )
    
    # マップを計算
    map1x, map1y = cv2.initUndistortRectifyMap(
        camera_matrix1, dist_coeffs1, R1, P1, img_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        camera_matrix2, dist_coeffs2, R2, P2, img_size, cv2.CV_32FC1
    )
    
    # 画像を校正
    rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    return rectified_img1, rectified_img2, Q


def _ncc_cost(
    left: np.ndarray,
    right: np.ndarray,
    x: int,
    y: int,
    d: int,
    patch_radius: int,
) -> float:
    """
    単一画素 (x, y) に対して、視差 d のときのNCCコストを計算
    コストは「1 - NCC」として定義（小さいほど良い）
    """
    h, w = left.shape
    x_r = x - d

    # 右画像側が範囲外なら大きなコスト
    if x_r - patch_radius < 0 or x_r + patch_radius >= w:
        return 1.0

    # パッチ範囲が画像内か確認
    if (
        x - patch_radius < 0
        or x + patch_radius >= w
        or y - patch_radius < 0
        or y + patch_radius >= h
    ):
        return 1.0

    patch_l = left[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1].astype(
        np.float32
    )
    patch_r = right[
        y - patch_radius : y + patch_radius + 1,
        x_r - patch_radius : x_r + patch_radius + 1,
    ].astype(np.float32)

    mean_l = patch_l.mean()
    mean_r = patch_r.mean()
    std_l = patch_l.std()
    std_r = patch_r.std()

    # 分散が小さすぎる場合はテクスチャが少ないのでコストを悪くする
    eps = 1e-5
    if std_l < eps or std_r < eps:
        return 1.0

    ncc = np.mean((patch_l - mean_l) * (patch_r - mean_r) / (std_l * std_r + eps))
    # NCCは[-1, 1]だが、コストは[0, 2]程度になる（1 - ncc）
    return float(1.0 - ncc)


def compute_disparity(
    rectified_img1: np.ndarray,
    rectified_img2: np.ndarray,
    max_disparity: int = 64,
    patch_radius: int = 3,
    iters: int = 3,
) -> np.ndarray:
    """
    PatchMatch + NCC による視差マップを計算

    Args:
        rectified_img1, rectified_img2: 校正された画像（左右画像）
        max_disparity: 探索する最大視差（0 〜 max_disparity-1）
        patch_radius: NCCを計算するパッチの半径（パッチサイズは (2r+1)^2）
        iters: PatchMatchの反復回数

    Returns:
        disparity: 視差マップ（float32）
    """
    # グレースケールに変換
    left = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    h, w = left.shape

    # 視差初期化（0〜max_disparity-1 の乱数）
    rng = np.random.default_rng()
    disparity = rng.integers(0, max_disparity, size=(h, w), dtype=np.int32)

    # コスト配列初期化
    cost = np.full((h, w), 1.0, dtype=np.float32)

    # 初期コスト計算
    for y in range(patch_radius, h - patch_radius):
        for x in range(patch_radius, w - patch_radius):
            d = int(disparity[y, x])
            cost[y, x] = _ncc_cost(left, right, x, y, d, patch_radius)

    # PatchMatch の反復
    for it in range(iters):
        # 偶数イテレーション: 左上→右下, 奇数: 右下→左上
        if it % 2 == 0:
            y_range = range(patch_radius, h - patch_radius)
            x_range = range(patch_radius, w - patch_radius)
            direction = 1
        else:
            y_range = range(h - patch_radius - 1, patch_radius - 1, -1)
            x_range = range(w - patch_radius - 1, patch_radius - 1, -1)
            direction = -1

        for y in y_range:
            for x in x_range:
                current_d = int(disparity[y, x])
                current_cost = cost[y, x]

                # 1. 近傍からの伝播（propagation）
                #   左右方向・上下方向の近傍を使う
                neighbors = []
                if 0 <= x - direction < w:
                    neighbors.append((y, x - direction))
                if 0 <= y - direction < h:
                    neighbors.append((y - direction, x))

                for ny, nx in neighbors:
                    nd = int(disparity[ny, nx])
                    if nd < 0 or nd >= max_disparity:
                        continue
                    c = _ncc_cost(left, right, x, y, nd, patch_radius)
                    if c < current_cost:
                        current_cost = c
                        current_d = nd

                # 2. ランダム探索（random search）
                radius = max_disparity // 2
                while radius >= 1:
                    # current_d 周辺のランダムサンプル
                    d_min = max(0, current_d - radius)
                    d_max = min(max_disparity - 1, current_d + radius)
                    rd = int(rng.integers(d_min, d_max + 1))
                    c = _ncc_cost(left, right, x, y, rd, patch_radius)
                    if c < current_cost:
                        current_cost = c
                        current_d = rd
                    radius //= 2

                disparity[y, x] = current_d
                cost[y, x] = current_cost

    # float32 に変換（そのまま視差値として返す）
    return disparity.astype(np.float32)


def disparity_to_depth(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    視差マップから深度マップを計算
    
    Args:
        disparity: 視差マップ
        Q: 再投影行列
    
    Returns:
        depth: 深度マップ（メートル単位）
    """
    # 視差から3D点を再投影
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Z座標が深度
    depth = points_3d[:, :, 2]
    
    # 無効な深度値を0に設定
    depth[depth <= 0] = 0
    depth[depth > 100] = 0  # 100m以上は無効とみなす
    
    return depth

