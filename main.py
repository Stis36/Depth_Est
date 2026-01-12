"""
Multi-View Stereo (MVS) Depth Estimation
複数のカメラ画像から深度を推定する
"""

import argparse
import numpy as np
from pathlib import Path

from modules.module_utils import load_mvs_images, setup_mvs_parameters, load_config
from modules.module_MVS import compute_mvs_depth
import cv2


def visualize_mvs_results(reference_img: np.ndarray, depth: np.ndarray, config: dict = None):
    """
    MVS結果を可視化
    
    Args:
        reference_img: 基準ビューの画像
        depth: 深度マップ
        config: 設定辞書
    """
    from pathlib import Path
    
    # デフォルト設定
    show_images = True
    save_results = True
    output_dir = "."
    save_depth = True
    
    # configから設定を読み込み
    if config is not None and 'output' in config:
        output_config = config['output']
        show_images = output_config.get('show_images', True)
        save_results = output_config.get('save_results', True)
        output_dir = output_config.get('output_dir', ".")
        save_depth = output_config.get('save_depth', True)
        
        # 出力ディレクトリを作成
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 深度マップを可視化
    depth_vis = depth.copy()
    depth_vis[depth_vis <= 0] = np.nan
    if np.any(~np.isnan(depth_vis)):
        depth_normalized = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    else:
        depth_colored = np.zeros_like(reference_img)
    
    # 結果を表示
    if show_images:
        cv2.imshow('Reference View', reference_img)
        cv2.imshow('Depth Map', depth_colored)
        
        print("キーを押すとウィンドウが閉じます...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 結果を保存
    if save_results:
        saved_files = []
        if save_depth:
            depth_path = str(Path(output_dir) / 'depth_map.png')
            cv2.imwrite(depth_path, depth_colored)
            saved_files.append(depth_path)
        
        if saved_files:
            print(f"結果を保存しました: {', '.join(saved_files)}")


def estimate_mvs_depth(config: dict = None):
    """
    MVS深度推定のメイン関数
    
    Args:
        config: 設定辞書（Noneの場合はconfig.yamlを読み込むかデフォルト値を使用）
    
    Returns:
        depth: 深度マップ
    """
    if config is None:
        raise ValueError("設定ファイルが必要です。")
    
    # num_viewsは参照ビュー以外のビュー数
    num_other_views = config.get('num_views', 2)
    if num_other_views < 1 or num_other_views > 3:
        raise ValueError(f"参照ビュー以外のビュー数は1-3の範囲である必要があります。現在: {num_other_views}")
    
    # 合計ビュー数 = 参照ビュー + 他のビュー
    total_views = num_other_views + 1
    
    # 参照ビューIDを取得
    reference_view_id = config.get('reference_view_id', 0)
    if reference_view_id < 0 or reference_view_id >= total_views:
        raise ValueError(f"参照ビューIDは0から{total_views-1}の範囲である必要があります。現在: {reference_view_id}")
    
    print(f"参照ビュー: ビュー {reference_view_id}")
    print(f"合計ビュー数: {total_views}（参照ビュー1つ + 他のビュー{num_other_views}つ）")
    
    # 画像パスを取得（すべてのビュー）
    image_paths = []
    for i in range(total_views):
        if i >= len(config['views']):
            raise ValueError(f"ビュー {i} の設定が見つかりません。")
        view = config['views'][i]
        img_path = view.get('image_path')
        if img_path is None:
            raise ValueError(f"ビュー {i} の画像パスが指定されていません。")
        image_paths.append(img_path)
    
    print(f"画像を読み込んでいます（合計{total_views}ビュー）...")
    images = load_mvs_images(image_paths)
    print(f"画像サイズ: {images[0].shape}")
    
    # 参照ビュー
    reference_img = images[reference_view_id]
    
    print("カメラパラメータを設定しています...")
    camera_matrices, dist_coeffs_list, R_list, T_list = setup_mvs_parameters(config, reference_view_id)
    
    # 参照ビュー以外のビューとパラメータ
    other_views = [img for i, img in enumerate(images) if i != reference_view_id]
    other_camera_matrices = [cam for i, cam in enumerate(camera_matrices) if i != reference_view_id]
    other_dist_coeffs = [dist for i, dist in enumerate(dist_coeffs_list) if i != reference_view_id]
    other_Rs = [R for i, R in enumerate(R_list) if i != reference_view_id]
    other_Ts = [T for i, T in enumerate(T_list) if i != reference_view_id]
    
    # MVS設定を取得
    mvs_config = config.get('mvs', {})
    depth_range = (
        mvs_config.get('depth_range', {}).get('min_depth', 0.1),
        mvs_config.get('depth_range', {}).get('max_depth', 10.0)
    )
    patch_radius = mvs_config.get('patch_radius', 3)
    iters = mvs_config.get('iters', 3)  # PatchMatchの反復回数
    
    print("MVS深度推定を実行しています（PatchMatch法）...")
    depth = compute_mvs_depth(
        reference_img,
        camera_matrices[0],
        dist_coeffs_list[0],
        other_views,
        other_camera_matrices,
        other_dist_coeffs,
        other_Rs,
        other_Ts,
        depth_range,
        patch_radius,
        iters,
    )
    
    print(f"深度範囲: {np.min(depth[depth > 0]):.2f}m - {np.max(depth[depth > 0]):.2f}m")
    
    print("結果を可視化しています...")
    visualize_mvs_results(reference_img, depth, config)
    
    return depth


def main():
    """
    メイン実行関数
    """
    parser = argparse.ArgumentParser(description='Multi-View Stereo Depth Estimation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # config.yamlを読み込む
    config = None
    if Path(args.config).exists():
        print(f"設定ファイルを読み込んでいます: {args.config}")
        config = load_config(args.config)
    else:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {args.config}")
    
    depth = estimate_mvs_depth(config)
    print("MVS深度推定が完了しました。")


if __name__ == "__main__":
    main()
