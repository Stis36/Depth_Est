"""
Utility functions: 画像読み込み、カメラパラメータ設定、可視化
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, List


def load_images(img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    2枚のカメラ画像を読み込む
    
    Args:
        img1_path: 1枚目の画像のパス
        img2_path: 2枚目の画像のパス
    
    Returns:
        img1, img2: 読み込まれた画像
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        raise ValueError(f"画像を読み込めませんでした: {img1_path}")
    if img2 is None:
        raise ValueError(f"画像を読み込めませんでした: {img2_path}")
    
    return img1, img2


def load_mvs_images(image_paths: List[str]) -> List[np.ndarray]:
    """
    複数のカメラ画像を読み込む（MVS用）
    
    Args:
        image_paths: 画像パスのリスト
    
    Returns:
        images: 読み込まれた画像のリスト
    """
    images = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"画像を読み込めませんでした: {img_path} (ビュー {i+1})")
        images.append(img)
    return images


def load_config(config_path: str) -> Dict[str, Any]:
    """
    config.yamlファイルを読み込む
    
    Args:
        config_path: config.yamlファイルのパス
    
    Returns:
        config: 設定辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    オイラー角から回転行列を計算
    
    Args:
        roll: ロール角（ラジアン）
        pitch: ピッチ角（ラジアン）
        yaw: ヨー角（ラジアン）
    
    Returns:
        R: 回転行列（3x3）
    """
    # 各軸周りの回転行列
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # ZYX順の回転
    R = R_z @ R_y @ R_x
    return R.astype(np.float32)


def setup_camera_parameters(config: Dict[str, Any] = None):
    """
    カメラの内部パラメータと外部パラメータを設定
    configが指定されている場合はconfig.yamlから読み込み、指定されていない場合はデフォルト値を使用
    
    Args:
        config: 設定辞書（Noneの場合はデフォルト値を使用）
    
    Returns:
        camera_matrix1, dist_coeffs1: カメラ1の内部パラメータ
        camera_matrix2, dist_coeffs2: カメラ2の内部パラメータ
        R, T: カメラ間の回転行列と並進ベクトル
    """
    if config is None:
        # デフォルト値（後方互換性のため）
        focal_length = 800.0
        cx, cy = 320.0, 240.0
        k1, k2, p1, p2, k3 = 0.1, -0.2, 0.0, 0.0, 0.0
        roll, pitch, yaw = 0.0, 0.0, 0.0
        tx, ty, tz = 0.10, 0.0, 0.0
    else:
        # カメラ1の内部パラメータ
        cam1_intrinsic = config['camera1']['intrinsic']
        focal_length = float(cam1_intrinsic['focal_length'])
        cx = float(cam1_intrinsic['cx'])
        cy = float(cam1_intrinsic['cy'])
        
        # カメラ1の歪み係数
        cam1_dist = config['camera1']['distortion']
        k1 = float(cam1_dist['k1'])
        k2 = float(cam1_dist['k2'])
        p1 = float(cam1_dist['p1'])
        p2 = float(cam1_dist['p2'])
        k3 = float(cam1_dist['k3'])
        
        # 外部パラメータ
        extrinsic = config['extrinsic']
        rotation_config = extrinsic['rotation']
        
        # 回転行列の初期化
        R = None
        roll = pitch = yaw = None
        
        if rotation_config['type'] == 'euler':
            roll = float(rotation_config['roll'])
            pitch = float(rotation_config['pitch'])
            yaw = float(rotation_config['yaw'])
        elif rotation_config['type'] == 'matrix':
            # matrix形式の場合
            R = np.array(rotation_config['matrix'], dtype=np.float32)
        
        translation = extrinsic['translation']
        tx = float(translation['x'])
        ty = float(translation['y'])
        tz = float(translation['z'])
    
    # カメラ1の内部パラメータ（カメラマトリックス）
    camera_matrix1 = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # カメラ1の歪み係数
    dist_coeffs1 = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    
    # カメラ2の内部パラメータ
    if config is not None:
        cam2_intrinsic = config['camera2']['intrinsic']
        focal_length2 = float(cam2_intrinsic['focal_length'])
        cx2 = float(cam2_intrinsic['cx'])
        cy2 = float(cam2_intrinsic['cy'])
        
        camera_matrix2 = np.array([
            [focal_length2, 0, cx2],
            [0, focal_length2, cy2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        cam2_dist = config['camera2']['distortion']
        k1_2 = float(cam2_dist['k1'])
        k2_2 = float(cam2_dist['k2'])
        p1_2 = float(cam2_dist['p1'])
        p2_2 = float(cam2_dist['p2'])
        k3_2 = float(cam2_dist['k3'])
        
        dist_coeffs2 = np.array([k1_2, k2_2, p1_2, p2_2, k3_2], dtype=np.float32)
    else:
        # デフォルト値: カメラ1と同じ
        camera_matrix2 = camera_matrix1.copy()
        dist_coeffs2 = dist_coeffs1.copy()
    
    # 外部パラメータ（回転行列）
    if R is None:
        if roll is not None:
            R = euler_to_rotation_matrix(roll, pitch, yaw)
        else:
            # デフォルト: 単位行列（並行配置）
            R = np.eye(3, dtype=np.float32)
    
    # 外部パラメータ（並進ベクトル）
    T = np.array([tx, ty, tz], dtype=np.float32)
    
    return camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T


def setup_mvs_parameters(config: Dict[str, Any], reference_view_id: int = 0) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    """
    MVS用のカメラパラメータを設定（複数ビュー対応）
    
    Args:
        config: 設定辞書
        reference_view_id: 参照ビューのID（デフォルト: 0）
    
    Returns:
        camera_matrices: 各ビューのカメラ内部パラメータリスト（参照ビューが最初）
        dist_coeffs_list: 各ビューの歪み係数リスト（参照ビューが最初）
        R_list: 参照ビューから各ビューへの回転行列リスト（参照ビューは単位行列）
        T_list: 参照ビューから各ビューへの並進ベクトルリスト（参照ビューはゼロベクトル）
    """
    # num_viewsは参照ビュー以外のビュー数
    num_other_views = config.get('num_views', 2)
    if num_other_views < 1 or num_other_views > 3:
        raise ValueError(f"参照ビュー以外のビュー数は1-3の範囲である必要があります。現在: {num_other_views}")
    
    # 合計ビュー数 = 参照ビュー + 他のビュー
    total_views = num_other_views + 1
    
    if reference_view_id < 0 or reference_view_id >= total_views:
        raise ValueError(f"参照ビューIDは0から{total_views-1}の範囲である必要があります。現在: {reference_view_id}")
    
    camera_matrices = []
    dist_coeffs_list = []
    R_list = []
    T_list = []
    
    # 参照ビューのパラメータ（最初に追加）
    ref_view = config['views'][reference_view_id]
    focal_length_ref = float(ref_view['intrinsic']['focal_length'])
    cx_ref = float(ref_view['intrinsic']['cx'])
    cy_ref = float(ref_view['intrinsic']['cy'])
    
    camera_matrix_ref = np.array([
        [focal_length_ref, 0, cx_ref],
        [0, focal_length_ref, cy_ref],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_ref = ref_view['distortion']
    dist_coeffs_ref = np.array([
        float(dist_ref['k1']), float(dist_ref['k2']), 
        float(dist_ref['p1']), float(dist_ref['p2']), float(dist_ref['k3'])
    ], dtype=np.float32)
    
    # 参照ビューは自分自身への変換なので単位行列とゼロベクトル
    R_ref = np.eye(3, dtype=np.float32)
    T_ref = np.zeros(3, dtype=np.float32)
    
    camera_matrices.append(camera_matrix_ref)
    dist_coeffs_list.append(dist_coeffs_ref)
    R_list.append(R_ref)
    T_list.append(T_ref)
    
    # 他のビューのパラメータ（参照ビュー以外）
    for i in range(total_views):
        if i == reference_view_id:
            continue
        view = config['views'][i]
        
        # 内部パラメータ
        focal_length = float(view['intrinsic']['focal_length'])
        cx = float(view['intrinsic']['cx'])
        cy = float(view['intrinsic']['cy'])
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 歪み係数
        dist = view['distortion']
        dist_coeffs = np.array([
            float(dist['k1']), float(dist['k2']),
            float(dist['p1']), float(dist['p2']), float(dist['k3'])
        ], dtype=np.float32)
        
        # 外部パラメータ（参照ビューから見た相対姿勢）
        extrinsic = view['extrinsic']
        rotation_config = extrinsic['rotation']
        
        R = None
        if rotation_config['type'] == 'euler':
            roll = float(rotation_config['roll'])
            pitch = float(rotation_config['pitch'])
            yaw = float(rotation_config['yaw'])
            R = euler_to_rotation_matrix(roll, pitch, yaw)
        elif rotation_config['type'] == 'matrix':
            R = np.array(rotation_config['matrix'], dtype=np.float32)
        else:
            R = np.eye(3, dtype=np.float32)
        
        translation = extrinsic['translation']
        T = np.array([
            float(translation['x']),
            float(translation['y']),
            float(translation['z'])
        ], dtype=np.float32)
        
        camera_matrices.append(camera_matrix)
        dist_coeffs_list.append(dist_coeffs)
        R_list.append(R)
        T_list.append(T)
    
    return camera_matrices, dist_coeffs_list, R_list, T_list


def visualize_results(img1: np.ndarray, img2: np.ndarray, 
                     rectified_img1: np.ndarray, rectified_img2: np.ndarray,
                     disparity: np.ndarray, depth: np.ndarray,
                     config: Dict[str, Any] = None):
    """
    結果を可視化
    
    Args:
        img1, img2: 元の画像
        rectified_img1, rectified_img2: 校正された画像
        disparity: 視差マップ
        depth: 深度マップ
        config: 設定辞書（Noneの場合はデフォルト設定を使用）
    """
    # デフォルト設定
    show_images = True
    save_results = True
    output_dir = "."
    save_rectified = True
    save_disparity = True
    save_depth = True
    
    # configから設定を読み込み
    if config is not None and 'output' in config:
        output_config = config['output']
        show_images = output_config.get('show_images', True)
        save_results = output_config.get('save_results', True)
        output_dir = output_config.get('output_dir', ".")
        save_rectified = output_config.get('save_rectified', True)
        save_disparity = output_config.get('save_disparity', True)
        save_depth = output_config.get('save_depth', True)
        
        # 出力ディレクトリを作成
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    # 視差マップを可視化（0-255の範囲に正規化）
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disparity_colored = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    
    # 深度マップを可視化
    depth_vis = depth.copy()
    depth_vis[depth_vis <= 0] = np.nan
    if np.any(~np.isnan(depth_vis)):
        depth_normalized = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    else:
        depth_colored = np.zeros_like(img1)
    
    # 結果を表示
    if show_images:
        cv2.imshow('Original Image 1', img1)
        cv2.imshow('Original Image 2', img2)
        cv2.imshow('Rectified Image 1', rectified_img1)
        cv2.imshow('Rectified Image 2', rectified_img2)
        cv2.imshow('Disparity Map', disparity_colored)
        cv2.imshow('Depth Map', depth_colored)
        
        print("キーを押すとウィンドウが閉じます...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 結果を保存
    if save_results:
        saved_files = []
        if save_rectified:
            rectified_path1 = str(Path(output_dir) / 'rectified_img1.png')
            rectified_path2 = str(Path(output_dir) / 'rectified_img2.png')
            cv2.imwrite(rectified_path1, rectified_img1)
            cv2.imwrite(rectified_path2, rectified_img2)
            saved_files.extend([rectified_path1, rectified_path2])
        
        if save_disparity:
            disparity_path = str(Path(output_dir) / 'disparity_map.png')
            cv2.imwrite(disparity_path, disparity_colored)
            saved_files.append(disparity_path)
        
        if save_depth:
            depth_path = str(Path(output_dir) / 'depth_map.png')
            cv2.imwrite(depth_path, depth_colored)
            saved_files.append(depth_path)
        
        if saved_files:
            print(f"結果を保存しました: {', '.join(saved_files)}")

