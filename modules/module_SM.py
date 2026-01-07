"""
Stereo Matching functions: ステレオマッチング関連の関数
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


def compute_disparity(rectified_img1: np.ndarray, rectified_img2: np.ndarray) -> np.ndarray:
    """
    視差マップを計算
    
    Args:
        rectified_img1, rectified_img2: 校正された画像
    
    Returns:
        disparity: 視差マップ
    """
    # グレースケールに変換
    gray1 = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY)
    
    # StereoSGBMを使用して視差を計算
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # 視差範囲（16の倍数である必要がある）
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    
    return disparity

