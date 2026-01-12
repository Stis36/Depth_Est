"""
Multi-View Stereo (MVS) functions: 複数ビューからの深度推定
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


def check_gpu_available() -> Tuple[bool, torch.device]:
    """
    GPUの使用可能状態を確認
    
    Returns:
        use_gpu: GPUが使用可能かどうか
        device: 使用するデバイス（cudaまたはcpu）
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPUを使用します: {torch.cuda.get_device_name(0)}")
        return True, device
    else:
        device = torch.device('cpu')
        print("GPUは使用できません。CPUで実行します。")
        return False, device


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


def _compute_ncc_cost_batch_gpu(
    reference_img_t: torch.Tensor,
    reference_camera_matrix_t: torch.Tensor,
    other_views_t: List[torch.Tensor],
    other_camera_matrices_t: List[torch.Tensor],
    other_Rs_t: List[torch.Tensor],
    other_Ts_t: List[torch.Tensor],
    coords: torch.Tensor,
    depths: torch.Tensor,
    patch_radius: int,
    device: torch.device,
) -> torch.Tensor:
    """
    GPU版: 複数の画素に対して並列にNCCコストを計算（ベクトル化）
    
    Args:
        reference_img_t: 参照ビューの画像テンソル [H, W]
        reference_camera_matrix_t: 参照ビューのカメラ内部パラメータテンソル [3, 3]
        other_views_t: 他のビューの画像テンソルリスト
        other_camera_matrices_t: 他のビューのカメラ内部パラメータテンソルリスト
        other_Rs_t: 他のビューへの回転行列テンソルリスト
        other_Ts_t: 他のビューへの並進ベクトルテンソルリスト
        coords: 画素座標テンソル [N, 2] (x, y)
        depths: 深度テンソル [N]
        patch_radius: パッチ半径
        device: デバイス
    
    Returns:
        costs: コストテンソル [N]
    """
    h, w = reference_img_t.shape
    N = coords.shape[0]
    eps = 1e-5
    patch_size = 2 * patch_radius + 1
    
    # 座標と深度を取得
    x_coords = coords[:, 0].long()
    y_coords = coords[:, 1].long()
    
    # 範囲チェックマスク
    valid_mask = (
        (x_coords >= patch_radius) & (x_coords < w - patch_radius) &
        (y_coords >= patch_radius) & (y_coords < h - patch_radius)
    )
    
    # 無効な画素のコストは1.0
    costs = torch.ones(N, dtype=torch.float32, device=device)
    
    if not valid_mask.any():
        return costs
    
    # 有効な画素のみ処理
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return costs
    
    x_valid = x_coords[valid_indices]
    y_valid = y_coords[valid_indices]
    depths_valid = depths[valid_indices]
    
    # 正規化座標を計算
    x_norm = (x_valid.float() - reference_camera_matrix_t[0, 2]) / reference_camera_matrix_t[0, 0]
    y_norm = (y_valid.float() - reference_camera_matrix_t[1, 2]) / reference_camera_matrix_t[1, 1]
    
    # 3D点を計算 [N_valid, 3]
    points_3d = torch.stack([
        x_norm * depths_valid,
        y_norm * depths_valid,
        depths_valid
    ], dim=1)
    
    # 参照ビューのパッチを抽出 [N_valid, patch_size, patch_size]
    patch_size = 2 * patch_radius + 1
    patches_ref = []
    for i in range(len(valid_indices)):
        y, x = y_valid[i].item(), x_valid[i].item()
        patch = reference_img_t[y - patch_radius:y + patch_radius + 1,
                                x - patch_radius:x + patch_radius + 1]
        patches_ref.append(patch)
    patches_ref = torch.stack(patches_ref, dim=0)
    
    # 参照パッチの平均と標準偏差
    mean_ref = patches_ref.mean(dim=(1, 2), keepdim=True)
    std_ref = patches_ref.std(dim=(1, 2), keepdim=True) + eps
    
    # 標準偏差が小さいパッチは無効
    valid_patch_mask = (std_ref.squeeze() > eps)
    
    # 各ビューでのコストを計算
    all_costs = []
    for other_view_t, cam_mat_t, R_t, T_t in zip(other_views_t, other_camera_matrices_t, other_Rs_t, other_Ts_t):
        # 3D点を他のビューに投影
        points_cam = torch.matmul(R_t, points_3d.unsqueeze(-1)).squeeze(-1) + T_t.unsqueeze(0)
        
        # 正規化座標
        z_valid = points_cam[:, 2]
        z_mask = z_valid > eps
        
        if not z_mask.any():
            continue
        
        x_norm_other = points_cam[:, 0] / z_valid
        y_norm_other = points_cam[:, 1] / z_valid
        
        # ピクセル座標
        points_2d = torch.matmul(cam_mat_t, torch.stack([
            x_norm_other, y_norm_other, torch.ones_like(x_norm_other)
        ], dim=0))
        
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
                patch = other_view_t[y_o - patch_radius:y_o + patch_radius + 1,
                                    x_o - patch_radius:x_o + patch_radius + 1]
                patches_other.append(patch)
            else:
                patches_other.append(torch.zeros(patch_size, patch_size, dtype=torch.float32, device=device))
        patches_other = torch.stack(patches_other, dim=0)
        
        # 他のビューのパッチの平均と標準偏差
        mean_other = patches_other.mean(dim=(1, 2), keepdim=True)
        std_other = patches_other.std(dim=(1, 2), keepdim=True) + eps
        
        # NCCを計算
        ncc = torch.sum(
            (patches_ref - mean_ref) * (patches_other - mean_other) / (std_ref * std_other),
            dim=(1, 2)
        ) / (patch_size * patch_size)
        
        # 有効な投影のみを考慮
        cost = 1.0 - ncc
        cost = torch.where(valid_proj_mask & valid_patch_mask.squeeze(), cost, torch.tensor(1.0, device=device))
        all_costs.append(cost)
    
    # 複数ビューのコストを統合（平均）
    if len(all_costs) > 0:
        valid_costs = torch.stack(all_costs, dim=0)  # [num_views, N_valid]
        # 無効なコスト（1.0）を除外して平均
        valid_mask_costs = valid_costs < 0.99  # [num_views, N_valid]
        # 各画素について、有効なビューのコストの平均を計算
        mean_costs = torch.ones(len(valid_indices), dtype=torch.float32, device=device)
        for j in range(len(valid_indices)):
            valid_for_pixel = valid_mask_costs[:, j]
            if valid_for_pixel.any():
                mean_costs[j] = valid_costs[valid_for_pixel, j].mean()
        costs[valid_indices] = mean_costs
    
    return costs


def _mvs_ncc_cost_gpu(
    reference_img_t: torch.Tensor,
    reference_camera_matrix: np.ndarray,
    other_views_t: List[torch.Tensor],
    other_camera_matrices: List[np.ndarray],
    other_Rs: List[np.ndarray],
    other_Ts: List[np.ndarray],
    x: int,
    y: int,
    depth: float,
    patch_radius: int,
    device: torch.device,
) -> float:
    """
    GPU版: 参照ビューの画素(x, y)に対して、深度depthのときのMVS NCCコストを計算
    （単一画素用、後方互換性のため保持）
    """
    h, w = reference_img_t.shape
    
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
    patch_ref = reference_img_t[y - patch_radius:y + patch_radius + 1,
                                x - patch_radius:x + patch_radius + 1]
    
    mean_ref = patch_ref.mean().item()
    std_ref = patch_ref.std().item()
    
    eps = 1e-5
    if std_ref < eps:
        return 1.0
    
    # 各ビューに投影してNCCコストを計算
    costs = []
    for other_view_t, cam_mat, R, T in zip(other_views_t, other_camera_matrices, other_Rs, other_Ts):
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
        patch_other = other_view_t[y_other - patch_radius:y_other + patch_radius + 1,
                                  x_other - patch_radius:x_other + patch_radius + 1]
        
        mean_other = patch_other.mean().item()
        std_other = patch_other.std().item()
        
        if std_other < eps:
            continue
        
        # NCCを計算（PyTorchテンソルで計算）
        ncc = torch.mean((patch_ref - mean_ref) * (patch_other - mean_other) / (std_ref * std_other + eps)).item()
        cost = 1.0 - ncc
        costs.append(cost)
    
    # 複数ビューのコストを統合（平均）
    if len(costs) > 0:
        return float(np.mean(costs))
    else:
        return 1.0


def compute_depth_patchmatch(
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
    GPUが使用可能な場合はGPU、そうでなければCPUで実行
    
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
    # GPU/CPUの確認
    use_gpu, device = check_gpu_available()
    
    h, w = reference_img.shape[:2]
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) if len(reference_img.shape) == 3 else reference_img
    gray_ref = gray_ref.astype(np.float32)
    
    # 他のビューもグレースケールに変換
    gray_others = []
    for view in other_views:
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) if len(view.shape) == 3 else view
        gray_others.append(gray.astype(np.float32))
    
    min_depth, max_depth = depth_range
    
    # GPU/CPUに応じてテンソルまたはNumPy配列を使用
    if use_gpu:
        # GPU版: PyTorchテンソルに変換
        gray_ref_t = torch.from_numpy(gray_ref).to(device)
        gray_others_t = [torch.from_numpy(gray).to(device) for gray in gray_others]
        
        # カメラパラメータをテンソルに変換
        reference_camera_matrix_t = torch.from_numpy(reference_camera_matrix).to(device)
        other_camera_matrices_t = [torch.from_numpy(cam_mat).to(device) for cam_mat in other_camera_matrices]
        other_Rs_t = [torch.from_numpy(R).to(device) for R in other_Rs]
        other_Ts_t = [torch.from_numpy(T).to(device) for T in other_Ts]
        
        # 深度を初期化（ランダム）
        rng = np.random.default_rng()
        depth_map = rng.uniform(min_depth, max_depth, size=(h, w)).astype(np.float32)
        depth_map_t = torch.from_numpy(depth_map).to(device)
        
        # コスト配列初期化
        cost_map_t = torch.full((h, w), 1.0, dtype=torch.float32, device=device)
        
        print(f"深度範囲: {min_depth:.2f}m - {max_depth:.2f}m, PatchMatch反復: {iters}回 (GPU)")
        
        # 初期コスト計算（並列化）
        print("初期コストを計算しています（GPU並列処理）...")
        valid_y, valid_x = torch.meshgrid(
            torch.arange(patch_radius, h - patch_radius, device=device),
            torch.arange(patch_radius, w - patch_radius, device=device),
            indexing='ij'
        )
        valid_coords = torch.stack([valid_x.flatten(), valid_y.flatten()], dim=1)
        valid_depths = depth_map_t[valid_y.flatten(), valid_x.flatten()]
        
        # バッチサイズで分割して処理（メモリ効率のため）
        batch_size = 10000
        for i in range(0, len(valid_coords), batch_size):
            end_idx = min(i + batch_size, len(valid_coords))
            batch_coords = valid_coords[i:end_idx]
            batch_depths = valid_depths[i:end_idx]
            
            batch_costs = _compute_ncc_cost_batch_gpu(
                gray_ref_t, reference_camera_matrix_t,
                gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                batch_coords, batch_depths, patch_radius, device
            )
            
            # コストマップに書き戻し
            cost_map_t[batch_coords[:, 1], batch_coords[:, 0]] = batch_costs
        
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
                    current_depth = depth_map_t[y, x].item()
                    current_cost = cost_map_t[y, x].item()
                    
                    # Propagation: 近傍画素から深度を伝播
                    if direction == 1:
                        neighbors = [(y, x - 1), (y - 1, x)]
                    else:
                        neighbors = [(y, x + 1), (y + 1, x)]
                    
                    for ny, nx in neighbors:
                        if (patch_radius <= ny < h - patch_radius and
                            patch_radius <= nx < w - patch_radius):
                            neighbor_depth = depth_map_t[ny, nx].item()
                            # ベクトル化関数を使用（単一画素でも効率的）
                            coords = torch.tensor([[x, y]], device=device)
                            depths = torch.tensor([neighbor_depth], device=device)
                            neighbor_cost = _compute_ncc_cost_batch_gpu(
                                gray_ref_t, reference_camera_matrix_t,
                                gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                                coords, depths, patch_radius, device
                            )[0].item()
                            
                            if neighbor_cost < current_cost:
                                current_depth = neighbor_depth
                                current_cost = neighbor_cost
                    
                    # Random search: ランダムに深度を探索
                    alpha = 0.5
                    search_radius = max_depth - min_depth
                    rng = np.random.default_rng()
                    
                    while search_radius > 0.01:
                        depth_candidate = current_depth + rng.uniform(-search_radius, search_radius)
                        depth_candidate = np.clip(depth_candidate, min_depth, max_depth)
                        
                        # ベクトル化関数を使用
                        coords = torch.tensor([[x, y]], device=device)
                        depths = torch.tensor([depth_candidate], device=device)
                        candidate_cost = _compute_ncc_cost_batch_gpu(
                            gray_ref_t, reference_camera_matrix_t,
                            gray_others_t, other_camera_matrices_t, other_Rs_t, other_Ts_t,
                            coords, depths, patch_radius, device
                        )[0].item()
                        
                        if candidate_cost < current_cost:
                            current_depth = depth_candidate
                            current_cost = candidate_cost
                        
                        search_radius *= alpha
                    
                    # 更新
                    depth_map_t[y, x] = current_depth
                    cost_map_t[y, x] = current_cost
        
        # GPUからCPUに戻す
        depth_map = depth_map_t.cpu().numpy()
    else:
        # CPU版: 元のNumPy実装
        min_depth, max_depth = depth_range
        
        # 深度を初期化（ランダム）
        rng = np.random.default_rng()
        depth_map = rng.uniform(min_depth, max_depth, size=(h, w)).astype(np.float32)
        
        # コスト配列初期化
        cost_map = np.full((h, w), 1.0, dtype=np.float32)
        
        print(f"深度範囲: {min_depth:.2f}m - {max_depth:.2f}m, PatchMatch反復: {iters}回 (CPU)")
        
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
                        neighbors = [(y, x - 1), (y - 1, x)]
                    else:
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
                    alpha = 0.5
                    search_radius = max_depth - min_depth
                    
                    while search_radius > 0.01:
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
    
    # PatchMatch法で深度推定
    depth_map = compute_depth_patchmatch(
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

