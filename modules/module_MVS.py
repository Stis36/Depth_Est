"""
Multi-View Stereo (MVS) functions: 複数ビューからの深度推定
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def project_point_to_view(point_3d: np.ndarray, 
                          camera_matrix: np.ndarray,
                          R: np.ndarray, 
                          T: np.ndarray) -> np.ndarray:
    """
    3D点を別のビューに投影
    
    Args:
        point_3d: 3D点 [X, Y, Z]
        camera_matrix: カメラ内部パラメータ (3x3)
        R: 回転行列 (3x3)
        T: 並進ベクトル (3x1)
    
    Returns:
        point_2d: 投影された2D点 [u, v]
    """
    # カメラ座標系に変換
    point_cam = R @ point_3d.reshape(3, 1) + T.reshape(3, 1)
    
    # 正規化座標
    if point_cam[2, 0] <= 0:
        return np.array([-1, -1])  # 無効
    
    x_norm = point_cam[0, 0] / point_cam[2, 0]
    y_norm = point_cam[1, 0] / point_cam[2, 0]
    
    # ピクセル座標に変換
    point_2d = camera_matrix @ np.array([[x_norm], [y_norm], [1.0]])
    return point_2d[:2, 0]


def _mvs_ncc_cost(
    reference_img: np.ndarray,
    reference_camera_matrix: np.ndarray,
    other_views: List[np.ndarray],
    other_camera_matrices: List[np.ndarray],
    other_Rs: List[np.ndarray],
    other_Ts: List[np.ndarray],
    x: int,
    y: int,
    depth: float,
    patch_radius: int,
) -> float:
    """
    参照ビューの画素(x, y)に対して、深度depthのときのMVS NCCコストを計算
    コストは「1 - NCC」として定義（小さいほど良い）
    
    Args:
        reference_img: 参照ビューの画像
        reference_camera_matrix: 参照ビューのカメラ内部パラメータ
        other_views: 他のビューの画像リスト
        other_camera_matrices: 他のビューのカメラ内部パラメータリスト
        other_Rs: 他のビューへの回転行列リスト
        other_Ts: 他のビューへの並進ベクトルリスト
        x, y: 参照ビューの画素座標
        depth: 深度候補
        patch_radius: パッチ半径
    
    Returns:
        cost: NCCコスト（複数ビューの平均）
    """
    h, w = reference_img.shape[:2]
    
    # 範囲チェック
    if (x - patch_radius < 0 or x + patch_radius >= w or
        y - patch_radius < 0 or y + patch_radius >= h):
        return 1.0
    
    # 参照ビューの画素を3D点に変換
    x_norm = (x - reference_camera_matrix[0, 2]) / reference_camera_matrix[0, 0]
    y_norm = (y - reference_camera_matrix[1, 2]) / reference_camera_matrix[1, 1]
    
    # 3D点（基準カメラ座標系）
    point_3d = np.array([x_norm * depth, y_norm * depth, depth])
    
    # 参照ビューのパッチ
    patch_ref = reference_img[y - patch_radius:y + patch_radius + 1,
                             x - patch_radius:x + patch_radius + 1]
    
    mean_ref = patch_ref.mean()
    std_ref = patch_ref.std()
    
    eps = 1e-5
    if std_ref < eps:
        return 1.0
    
    # 各ビューに投影してNCCコストを計算
    costs = []
    for other_view, cam_mat, R, T in zip(other_views, other_camera_matrices, other_Rs, other_Ts):
        # 3D点を他のビューに投影
        point_2d = project_point_to_view(point_3d, cam_mat, R, T)
        
        if point_2d[0] < 0 or point_2d[1] < 0:
            continue
        
        x_other = int(round(point_2d[0]))
        y_other = int(round(point_2d[1]))
        
        # 範囲チェック
        if (x_other - patch_radius < 0 or x_other + patch_radius >= w or
            y_other - patch_radius < 0 or y_other + patch_radius >= h):
            continue
        
        # 他のビューのパッチ
        patch_other = other_view[y_other - patch_radius:y_other + patch_radius + 1,
                                 x_other - patch_radius:x_other + patch_radius + 1]
        
        mean_other = patch_other.mean()
        std_other = patch_other.std()
        
        if std_other < eps:
            continue
        
        # NCCを計算
        ncc = np.mean((patch_ref - mean_ref) * (patch_other - mean_other) / (std_ref * std_other + eps))
        cost = 1.0 - ncc
        costs.append(cost)
    
    # 複数ビューのコストを統合（平均）
    if len(costs) > 0:
        return float(np.mean(costs))
    else:
        return 1.0


def compute_depth_from_multiple_views(
    reference_img: np.ndarray,
    reference_camera_matrix: np.ndarray,
    other_views: List[np.ndarray],
    other_camera_matrices: List[np.ndarray],
    other_Rs: List[np.ndarray],
    other_Ts: List[np.ndarray],
    depth_range: Tuple[float, float] = (0.1, 10.0),
    patch_radius: int = 3,
    iters: int = 3,
) -> np.ndarray:
    """
    複数ビューから深度マップを計算（PatchMatch法）
    
    Args:
        reference_img: 基準ビューの画像
        reference_camera_matrix: 基準ビューのカメラ内部パラメータ
        other_views: 他のビューの画像リスト
        other_camera_matrices: 他のビューのカメラ内部パラメータリスト
        other_Rs: 他のビューへの回転行列リスト
        other_Ts: 他のビューへの並進ベクトルリスト
        depth_range: 深度の探索範囲 (min_depth, max_depth)
        patch_radius: NCC計算のパッチ半径
        iters: PatchMatchの反復回数
    
    Returns:
        depth_map: 深度マップ
    """
    h, w = reference_img.shape[:2]
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) if len(reference_img.shape) == 3 else reference_img
    gray_ref = gray_ref.astype(np.float32)
    
    # 他のビューもグレースケールに変換
    gray_others = []
    for view in other_views:
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) if len(view.shape) == 3 else view
        gray_others.append(gray.astype(np.float32))
    
    min_depth, max_depth = depth_range
    
    # 深度を初期化（ランダム）
    rng = np.random.default_rng()
    depth_map = rng.uniform(min_depth, max_depth, size=(h, w)).astype(np.float32)
    
    # コスト配列初期化
    cost_map = np.full((h, w), 1.0, dtype=np.float32)
    
    print(f"深度範囲: {min_depth:.2f}m - {max_depth:.2f}m, PatchMatch反復: {iters}回")
    
    # 初期コスト計算
    print("初期コストを計算しています...")
    for y in range(patch_radius, h - patch_radius):
        for x in range(patch_radius, w - patch_radius):
            depth = depth_map[y, x]
            cost_map[y, x] = _mvs_ncc_cost(
                gray_ref, reference_camera_matrix,
                gray_others, other_camera_matrices, other_Rs, other_Ts,
                x, y, depth, patch_radius
            )
    
    # PatchMatch の反復
    for it in range(iters):
        print(f"PatchMatch反復 {it + 1}/{iters} を実行中...")
        
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
                current_depth = depth_map[y, x]
                current_cost = cost_map[y, x]
                
                # Propagation: 近傍画素から深度を伝播
                if direction == 1:
                    # 左上→右下: 左と上の画素から伝播
                    neighbors = [(y, x - 1), (y - 1, x)]
                else:
                    # 右下→左上: 右と下の画素から伝播
                    neighbors = [(y, x + 1), (y + 1, x)]
                
                for ny, nx in neighbors:
                    if (patch_radius <= ny < h - patch_radius and
                        patch_radius <= nx < w - patch_radius):
                        neighbor_depth = depth_map[ny, nx]
                        neighbor_cost = _mvs_ncc_cost(
                            gray_ref, reference_camera_matrix,
                            gray_others, other_camera_matrices, other_Rs, other_Ts,
                            x, y, neighbor_depth, patch_radius
                        )
                        
                        if neighbor_cost < current_cost:
                            current_depth = neighbor_depth
                            current_cost = neighbor_cost
                
                # Random search: ランダムに深度を探索
                alpha = 0.5  # 探索範囲の縮小率
                search_radius = max_depth - min_depth
                
                while search_radius > 0.01:  # 1cm以下になったら終了
                    # 現在の深度を中心にランダム探索
                    depth_candidate = current_depth + rng.uniform(-search_radius, search_radius)
                    depth_candidate = np.clip(depth_candidate, min_depth, max_depth)
                    
                    candidate_cost = _mvs_ncc_cost(
                        gray_ref, reference_camera_matrix,
                        gray_others, other_camera_matrices, other_Rs, other_Ts,
                        x, y, depth_candidate, patch_radius
                    )
                    
                    if candidate_cost < current_cost:
                        current_depth = depth_candidate
                        current_cost = candidate_cost
                    
                    search_radius *= alpha
                
                # 更新
                depth_map[y, x] = current_depth
                cost_map[y, x] = current_cost
    
    return depth_map


def compute_mvs_depth(
    reference_img: np.ndarray,
    reference_camera_matrix: np.ndarray,
    reference_dist_coeffs: np.ndarray,
    other_views: List[np.ndarray],
    other_camera_matrices: List[np.ndarray],
    other_dist_coeffs: List[np.ndarray],
    other_Rs: List[np.ndarray],
    other_Ts: List[np.ndarray],
    depth_range: Tuple[float, float] = (0.1, 10.0),
    patch_radius: int = 3,
    iters: int = 3,
) -> np.ndarray:
    """
    MVS深度推定のメイン関数（歪み補正を含む、PatchMatch法）
    
    Args:
        reference_img: 基準ビューの画像
        reference_camera_matrix: 基準ビューのカメラ内部パラメータ
        reference_dist_coeffs: 基準ビューの歪み係数
        other_views: 他のビューの画像リスト
        other_camera_matrices: 他のビューのカメラ内部パラメータリスト
        other_dist_coeffs: 他のビューの歪み係数リスト
        other_Rs: 他のビューへの回転行列リスト
        other_Ts: 他のビューへの並進ベクトルリスト
        depth_range: 深度の探索範囲
        patch_radius: NCC計算のパッチ半径
        iters: PatchMatchの反復回数
    
    Returns:
        depth_map: 深度マップ
    """
    # 歪み補正
    h, w = reference_img.shape[:2]
    ref_undistorted = cv2.undistort(reference_img, reference_camera_matrix, reference_dist_coeffs)
    
    other_undistorted = []
    for view, cam_mat, dist_coeffs in zip(other_views, other_camera_matrices, other_dist_coeffs):
        undistorted = cv2.undistort(view, cam_mat, dist_coeffs)
        other_undistorted.append(undistorted)
    
    # 深度推定（PatchMatch法）
    depth_map = compute_depth_from_multiple_views(
        ref_undistorted,
        reference_camera_matrix,
        other_undistorted,
        other_camera_matrices,
        other_Rs,
        other_Ts,
        depth_range,
        patch_radius,
        iters,
    )
    
    return depth_map

