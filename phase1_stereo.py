import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 & 그레이스케일 변환
img_L = cv2.imread('data/Jadeplant/im0.png')
img_R = cv2.imread('data/Jadeplant/im1.png')
gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

# PFM 읽기
def read_pfm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline().decode().strip()
        dims = f.readline().decode().strip().split()
        W, H = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((H, W))
        if scale < 0:
            data = data[::-1]
    return data

gt_disp = read_pfm('data/Jadeplant/disp0.pfm')

# EPE 계산 (inf 마스킹)
def compute_epe(pred, gt):
    mask = np.isfinite(gt) & (gt > 0)
    return np.mean(np.abs(pred[mask] - gt[mask]))

# StereoBM
print("StereoBM 계산 중...")
bm = cv2.StereoBM_create(numDisparities=640, blockSize=15)
disp_bm = bm.compute(gray_L, gray_R).astype(np.float32) / 16.0

# StereoSGBM
print("StereoSGBM 계산 중...")
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=640,
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disp_sgbm = sgbm.compute(img_L, img_R).astype(np.float32) / 16.0

# EPE 출력
epe_bm   = compute_epe(disp_bm, gt_disp)
epe_sgbm = compute_epe(disp_sgbm, gt_disp)
print(f"StereoBM   EPE: {epe_bm:.2f} px")
print(f"StereoSGBM EPE: {epe_sgbm:.2f} px")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(gt_disp, cmap='plasma', vmin=0, vmax=600)
axes[0].set_title('GT Disparity')
axes[1].imshow(disp_bm, cmap='plasma', vmin=0, vmax=600)
axes[1].set_title(f'StereoBM (EPE={epe_bm:.1f}px)')
axes[2].imshow(disp_sgbm, cmap='plasma', vmin=0, vmax=600)
axes[2].set_title(f'StereoSGBM (EPE={epe_sgbm:.1f}px)')
plt.tight_layout()
plt.savefig('output_phase1_stereo.png', dpi=100)
plt.show()
print("저장 완료: output_phase1_stereo.png")