"""
Depth Estimation using Stereo Vision
2枚のカメラ画像から深度を推定する
"""

import argparse
import numpy as np
from pathlib import Path

from modules.module_utils import load_images, setup_camera_parameters, visualize_results, load_config
from modules.module_SM import stereo_rectify, compute_disparity, disparity_to_depth


def estimate_depth(img1_path: str = None, img2_path: str = None, config: dict = None):
    """
    メイン関数: 2枚のカメラ画像から深度を推定
    
    Args:
        img1_path: 1枚目の画像のパス（Noneの場合はconfigから読み込み）
        img2_path: 2枚目の画像のパス（Noneの場合はconfigから読み込み）
        config: 設定辞書（Noneの場合はconfig.yamlを読み込むかデフォルト値を使用）
    
    Returns:
        depth: 深度マップ
    """
    # 設定の読み込み
    if config is None:
        config = {}
    
    # 画像パスの取得（引数が優先、なければconfigから）
    if img1_path is None:
        if 'input' in config and 'img1_path' in config['input']:
            img1_path = config['input']['img1_path']
        else:
            raise ValueError("画像1のパスが指定されていません。--img1オプションまたはconfig.yamlで指定してください。")
    
    if img2_path is None:
        if 'input' in config and 'img2_path' in config['input']:
            img2_path = config['input']['img2_path']
        else:
            raise ValueError("画像2のパスが指定されていません。--img2オプションまたはconfig.yamlで指定してください。")
    
    print("画像を読み込んでいます...")
    img1, img2 = load_images(img1_path, img2_path)
    print(f"画像サイズ: {img1.shape}")
    
    print("カメラパラメータを設定しています...")
    camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T = setup_camera_parameters(config if config else None)
    
    print("ステレオ校正を実行しています...")
    rectified_img1, rectified_img2, Q = stereo_rectify(
        img1, img2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T
    )
    
    print("視差マップを計算しています...")
    disparity = compute_disparity(rectified_img1, rectified_img2)
    
    print("深度マップを計算しています...")
    depth = disparity_to_depth(disparity, Q)
    
    print(f"深度範囲: {np.min(depth[depth > 0]):.2f}m - {np.max(depth[depth > 0]):.2f}m")
    
    print("結果を可視化しています...")
    visualize_results(img1, img2, rectified_img1, rectified_img2, disparity, depth, config if config else None)
    
    return depth


def main():
    """
    メイン実行関数
    """
    parser = argparse.ArgumentParser(description='Stereo Vision Depth Estimation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to config.yaml file')
    parser.add_argument('--img1', type=str, default=None, 
                       help='Path to first camera image (overrides config)')
    parser.add_argument('--img2', type=str, default=None, 
                       help='Path to second camera image (overrides config)')
    
    args = parser.parse_args()
    
    # config.yamlを読み込む
    config = None
    if Path(args.config).exists():
        print(f"設定ファイルを読み込んでいます: {args.config}")
        config = load_config(args.config)
    else:
        print(f"警告: 設定ファイルが見つかりません: {args.config}。デフォルト設定を使用します。")
    
    depth = estimate_depth(args.img1, args.img2, config)
    print("深度推定が完了しました。")


if __name__ == "__main__":
    main()
