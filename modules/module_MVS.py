"""
Multi-View Stereo (MVS) functions: 複数ビューからの深度推定
シンプルなGPU実装のPatchMatch
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple


def check_gpu_available() -> Tuple[bool, torch.device]:
    """GPUの使用可能状態を確認"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPUを使用します: {torch.cuda.get_device_name(0)}")
        return True, device
    else:
        device = torch.device('cpu')
        print("GPUは使用できません。CPUで実行します。")
        return False, device


def compute_ncc_cost_gpu(
    ref_img: torch.Tensor,
    ref_cam: torch.Tensor,
    other_imgs: List[torch.Tensor],
    other_cams: List[torch.Tensor],
    other_Rs: List[torch.Tensor],
    other_Ts: List[torch.Tensor],
    coords: torch.Tensor,  # [N, 2] (x, y)
    depths: torch.Tensor,  # [N]
    patch_radius: int,
    device: torch.device,
) -> torch.Tensor:
    """
    シンプルなGPU版NCCコスト計算（全画素並列処理）
    
    Args:
        ref_img: 参照画像 [H, W]
        ref_cam: 参照カメラ内部パラメータ [3, 3]
        other_imgs: 他のビューの画像リスト
        other_cams: 他のビューのカメラ内部パラメータリスト
        other_Rs: 回転行列リスト
        other_Ts: 並進ベクトルリスト
        coords: 画素座標 [N, 2]
        depths: 深度 [N]
        patch_radius: パッチ半径
        device: デバイス
    
    Returns:
        costs: コスト [N]
    """
    h, w = ref_img.shape
    N = coords.shape[0]
    patch_size = 2 * patch_radius + 1
    eps = 1e-5
    
    x_coords = coords[:, 0].long()
    y_coords = coords[:, 1].long()
    
    # 範囲チェック
    valid_mask = (
        (x_coords >= patch_radius) & (x_coords < w - patch_radius) &
        (y_coords >= patch_radius) & (y_coords < h - patch_radius)
    )
    
    costs = torch.ones(N, dtype=torch.float32, device=device)
    
    if not valid_mask.any():
        return costs
    
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return costs
    
    x_valid = x_coords[valid_indices]
    y_valid = y_coords[valid_indices]
    depths_valid = depths[valid_indices]
    
    # 正規化座標
    x_norm = (x_valid.float() - ref_cam[0, 2]) / ref_cam[0, 0]
    y_norm = (y_valid.float() - ref_cam[1, 2]) / ref_cam[1, 1]
    
    # 3D点 [N_valid, 3]
    points_3d = torch.stack([
        x_norm * depths_valid,
        y_norm * depths_valid,
        depths_valid
    ], dim=1)
    
    # 参照パッチを抽出 [N_valid, patch_size, patch_size]
    # ベクトル化されたパッチ抽出
    patches_ref = []
    for i in range(len(valid_indices)):
        y, x = y_valid[i].item(), x_valid[i].item()
        patch = ref_img[y - patch_radius:y + patch_radius + 1,
                       x - patch_radius:x + patch_radius + 1]
        patches_ref.append(patch)
    patches_ref = torch.stack(patches_ref, dim=0)
    
    mean_ref = patches_ref.mean(dim=(1, 2), keepdim=True)
    std_ref = patches_ref.std(dim=(1, 2), keepdim=True) + eps
    
    valid_patch_mask = (std_ref.squeeze() > eps)
    
    # 各ビューでのコストを計算
    all_costs = []
    for other_img, cam_mat, R, T in zip(other_imgs, other_cams, other_Rs, other_Ts):
        # 3D点を他のビューに投影
        points_cam = torch.matmul(R, points_3d.unsqueeze(-1)).squeeze(-1) + T.unsqueeze(0)
        
        z_valid = points_cam[:, 2]
        z_mask = z_valid > eps
        
        if not z_mask.any():
            continue
        
        x_norm_other = points_cam[:, 0] / z_valid
        y_norm_other = points_cam[:, 1] / z_valid
        
        # ピクセル座標
        ones = torch.ones_like(x_norm_other)
        points_2d = torch.matmul(cam_mat, torch.stack([x_norm_other, y_norm_other, ones], dim=0))
        
        x_other = points_2d[0, :].round().long()
        y_other = points_2d[1, :].round().long()
        
        # 範囲チェック
        valid_proj_mask = (
            z_mask &
            (x_other >= patch_radius) & (x_other < w - patch_radius) &
            (y_other >= patch_radius) & (y_other < h - patch_radius)
        )
        
        if not valid_proj_mask.any():
            continue
        
        # 他のビューのパッチを抽出
        patches_other = []
        for i in range(len(valid_indices)):
            if valid_proj_mask[i]:
                y_o, x_o = y_other[i].item(), x_other[i].item()
                patch = other_img[y_o - patch_radius:y_o + patch_radius + 1,
                                 x_o - patch_radius:x_o + patch_radius + 1]
                patches_other.append(patch)
            else:
                patches_other.append(torch.zeros(patch_size, patch_size, dtype=torch.float32, device=device))
        patches_other = torch.stack(patches_other, dim=0)
        
        mean_other = patches_other.mean(dim=(1, 2), keepdim=True)
        std_other = patches_other.std(dim=(1, 2), keepdim=True) + eps
        
        # NCC計算
        ncc = torch.sum(
            (patches_ref - mean_ref) * (patches_other - mean_other) / (std_ref * std_other),
            dim=(1, 2)
        ) / (patch_size * patch_size)
        
        cost = 1.0 - ncc
        cost = torch.where(valid_proj_mask & valid_patch_mask.squeeze(), cost, torch.tensor(1.0, device=device))
        all_costs.append(cost)
    
    # 複数ビューのコストを平均
    if len(all_costs) > 0:
        valid_costs = torch.stack(all_costs, dim=0)  # [num_views, N_valid]
        # 無効なコスト（1.0）を除外して平均
        valid_mask_costs = valid_costs < 0.99
        mean_costs = torch.ones(len(valid_indices), dtype=torch.float32, device=device)
        for j in range(len(valid_indices)):
            valid_for_pixel = valid_mask_costs[:, j]
            if valid_for_pixel.any():
                mean_costs[j] = valid_costs[valid_for_pixel, j].mean()
        costs[valid_indices] = mean_costs
    
    return costs


def compute_depth_patchmatch_gpu(
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
    シンプルなGPU版PatchMatch深度推定
    
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
    use_gpu, device = check_gpu_available()
    
    if not use_gpu:
        raise RuntimeError("この実装はGPU専用です。GPUが使用できない場合は別の実装を使用してください。")
    
    h, w = reference_img.shape[:2]
    
    # グレースケール変換
    if len(reference_img.shape) == 3:
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_ref = reference_img.astype(np.float32)
    
    gray_others = []
    for view in other_views:
        if len(view.shape) == 3:
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = view.astype(np.float32)
        gray_others.append(gray)
    
    # GPUテンソルに変換
    gray_ref_t = torch.from_numpy(gray_ref).to(device)
    gray_others_t = [torch.from_numpy(gray).to(device) for gray in gray_others]
    
    reference_camera_matrix_t = torch.from_numpy(reference_camera_matrix).to(device)
    other_camera_matrices_t = [torch.from_numpy(cam_mat).to(device) for cam_mat in other_camera_matrices]
    other_Rs_t = [torch.from_numpy(R).to(device) for R in other_Rs]
    other_Ts_t = [torch.from_numpy(T).to(device) for T in other_Ts]
    
    min_depth, max_depth = depth_range
    
    # 深度を初期化（ランダム）
    rng = np.random.default_rng()
    depth_map = rng.uniform(min_depth, max_depth, size=(h, w)).astype(np.float32)
    depth_map_t = torch.from_numpy(depth_map).to(device)
    
    # コスト配列初期化
    cost_map_t = torch.full((h, w), 1.0, dtype=torch.float32, device=device)
    
    print(f"深度範囲: {min_depth:.2f}m - {max_depth:.2f}m, PatchMatch反復: {iters}回 (GPU)")
    
    # 有効な画素座標を取得
    valid_y, valid_x = torch.meshgrid(
        torch.arange(patch_radius, h - patch_radius, device=device),
        torch.arange(patch_radius, w - patch_radius, device=device),
        indexing='ij'
    )
    valid_coords = torch.stack([valid_x.flatten(), valid_y.flatten()], dim=1)
    
    # 初期コスト計算（全画素並列処理）
    print("初期コストを計算しています（GPU並列処理）...")
    valid_depths = depth_map_t[valid_y.flatten(), valid_x.flatten()]
    
    # バッチサイズで分割（メモリ効率のため）
    batch_size = 50000
    for i in range(0, len(valid_coords), batch_size):
        end_idx = min(i + batch_size, len(valid_coords))
        batch_coords = valid_coords[i:end_idx]
        batch_depths = valid_depths[i:end_idx]
        
        batch_costs = compute_ncc_cost_gpu(
            gray_ref_t, reference_camera_matrix_t,
            gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
            batch_coords, batch_depths, patch_radius, device
        )
        
        cost_map_t[batch_coords[:, 1], batch_coords[:, 0]] = batch_costs
    
    # PatchMatch反復
    for it in range(iters):
        print(f"PatchMatch反復 {it + 1}/{iters} を実行中...")
        
        # スキャン順序（偶数: 左上→右下, 奇数: 右下→左上）
        if it % 2 == 0:
            y_range = range(patch_radius, h - patch_radius)
            x_range = range(patch_radius, w - patch_radius)
            direction = 1
        else:
            y_range = range(h - patch_radius - 1, patch_radius - 1, -1)
            x_range = range(w - patch_radius - 1, patch_radius - 1, -1)
            direction = -1
        
        # 各行を並列処理
        for y in y_range:
            # 行内の全画素を一度に処理
            row_x = torch.arange(patch_radius, w - patch_radius, device=device)
            row_coords = torch.stack([
                row_x,
                torch.full((len(row_x),), y, dtype=torch.long, device=device)
            ], dim=1)
            
            row_depths = depth_map_t[y, row_x]
            
            # 現在のコストを計算
            row_costs = compute_ncc_cost_gpu(
                gray_ref_t, reference_camera_matrix_t,
                gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                row_coords, row_depths, patch_radius, device
            )
            
            # Propagation: 近傍から深度を伝播
            if direction == 1:
                # 左と上から伝播
                if x_range.start > patch_radius:
                    left_depths = depth_map_t[y, row_x - 1]
                    left_costs = compute_ncc_cost_gpu(
                        gray_ref_t, reference_camera_matrix_t,
                        gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                        row_coords, left_depths, patch_radius, device
                    )
                    better_mask = left_costs < row_costs
                    row_depths = torch.where(better_mask, left_depths, row_depths)
                    row_costs = torch.where(better_mask, left_costs, row_costs)
                
                if y > patch_radius:
                    up_coords = torch.stack([
                        row_x,
                        torch.full((len(row_x),), y - 1, dtype=torch.long, device=device)
                    ], dim=1)
                    up_depths = depth_map_t[y - 1, row_x]
                    up_costs = compute_ncc_cost_gpu(
                        gray_ref_t, reference_camera_matrix_t,
                        gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                        up_coords, up_depths, patch_radius, device
                    )
                    better_mask = up_costs < row_costs
                    row_depths = torch.where(better_mask, up_depths, row_depths)
                    row_costs = torch.where(better_mask, up_costs, row_costs)
            else:
                # 右と下から伝播
                if x_range.stop < w - patch_radius:
                    right_depths = depth_map_t[y, row_x + 1]
                    right_costs = compute_ncc_cost_gpu(
                        gray_ref_t, reference_camera_matrix_t,
                        gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                        row_coords, right_depths, patch_radius, device
                    )
                    better_mask = right_costs < row_costs
                    row_depths = torch.where(better_mask, right_depths, row_depths)
                    row_costs = torch.where(better_mask, right_costs, row_costs)
                
                if y < h - patch_radius - 1:
                    down_coords = torch.stack([
                        row_x,
                        torch.full((len(row_x),), y + 1, dtype=torch.long, device=device)
                    ], dim=1)
                    down_depths = depth_map_t[y + 1, row_x]
                    down_costs = compute_ncc_cost_gpu(
                        gray_ref_t, reference_camera_matrix_t,
                        gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                        down_coords, down_depths, patch_radius, device
                    )
                    better_mask = down_costs < row_costs
                    row_depths = torch.where(better_mask, down_depths, row_depths)
                    row_costs = torch.where(better_mask, down_costs, row_costs)
            
            # Random search: 各画素に対してランダム探索
            alpha = 0.5
            search_radius = max_depth - min_depth
            
            # ランダムな深度候補を生成
            random_offsets = torch.rand(len(row_x), device=device) * 2.0 - 1.0  # [-1, 1]
            depth_candidates = row_depths + random_offsets * search_radius
            depth_candidates = torch.clamp(depth_candidates, min_depth, max_depth)
            
            candidate_costs = compute_ncc_cost_gpu(
                gray_ref_t, reference_camera_matrix_t,
                gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                row_coords, depth_candidates, patch_radius, device
            )
            
            better_mask = candidate_costs < row_costs
            row_depths = torch.where(better_mask, depth_candidates, row_depths)
            row_costs = torch.where(better_mask, candidate_costs, row_costs)
            
            # 更新
            depth_map_t[y, row_x] = row_depths
            cost_map_t[y, row_x] = row_costs
    
    # GPUからCPUに戻す
    depth_map = depth_map_t.cpu().numpy()
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
    MVS深度推定のメイン関数（シンプルなGPU実装）
    
    Args:
        reference_img: 基準ビューの画像
        reference_camera_matrix: 基準ビューのカメラ内部パラメータ
        reference_dist_coeffs: 基準ビューの歪み係数（未使用、互換性のため保持）
        other_views: 他のビューの画像リスト
        other_camera_matrices: 他のビューのカメラ内部パラメータリスト
        other_dist_coeffs: 他のビューの歪み係数リスト（未使用、互換性のため保持）
        other_Rs: 他のビューへの回転行列リスト
        other_Ts: 他のビューへの並進ベクトルリスト
        depth_range: 深度の探索範囲
        patch_radius: NCC計算のパッチ半径
        iters: PatchMatchの反復回数
    
    Returns:
        depth_map: 深度マップ
    """
    # シンプルなGPU版PatchMatchで深度推定
    depth_map = compute_depth_patchmatch_gpu(
        reference_img,
        reference_camera_matrix,
        other_views,
        other_camera_matrices,
        other_Rs,
        other_Ts,
        depth_range,
        patch_radius,
        iters,
    )
    
    return depth_map
