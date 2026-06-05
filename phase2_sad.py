import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 이미지 로드 & 다운샘플링 (1/4 크기)
img_L = cv2.imread('data/Jadeplant/im0.png')
img_R = cv2.imread('data/Jadeplant/im1.png')

scale = 0.25
img_L_small = cv2.resize(img_L, None, fx=scale, fy=scale)
img_R_small = cv2.resize(img_R, None, fx=scale, fy=scale)
gray_L = cv2.cvtColor(img_L_small, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R_small, cv2.COLOR_BGR2GRAY)

print(f"처리 크기: {gray_L.shape}")  # 약 497 x 658

# PFM 읽기 & 동일 크기로 다운샘플
def read_pfm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline().decode().strip()
        dims = f.readline().decode().strip().split()
        W, H = int(dims[0]), int(dims[1])
        scale_pfm = float(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((H, W))
        if scale_pfm < 0:
            data = data[::-1]
    return data

gt_disp_full = read_pfm('data/Jadeplant/disp0.pfm')
# disparity도 스케일에 맞게 줄이기 (값도 scale 곱)
gt_disp = cv2.resize(gt_disp_full, None, fx=scale, fy=scale) * scale
gt_disp[~np.isfinite(gt_disp)] = 0  # inf 제거

# EPE 계산
def compute_epe(pred, gt):
    mask = (gt > 0)
    return np.mean(np.abs(pred[mask] - gt[mask]))

# SAD Cost Volume 구현
def compute_sad_cost_volume(img_l, img_r, max_disp=64, win=7):
    H, W = img_l.shape
    cost_vol = np.full((H, W, max_disp), np.inf)

    print(f"SAD Cost Volume 계산 중... (max_disp={max_disp})")
    start = time.time()

    for d in range(max_disp):
        # d만큼 오른쪽 이미지를 시프트
        shifted = np.zeros_like(img_r)
        if d == 0:
            shifted = img_r.copy()
        else:
            shifted[:, d:] = img_r[:, :W-d]

        # SAD: 절대 차이를 window 합산
        diff = np.abs(img_l.astype(np.float32) - shifted.astype(np.float32))
        cost_vol[:, :, d] = cv2.boxFilter(diff, -1, (win, win))

        if d % 16 == 0:
            elapsed = time.time() - start
            print(f"  d={d}/{max_disp} ({elapsed:.1f}s)")

    # Winner-Takes-All: 최소 cost의 disparity 선택
    disp_map = np.argmin(cost_vol, axis=2).astype(np.float32)
    print(f"완료! 총 {time.time()-start:.1f}s")
    return disp_map

# max_disp=160 (풀 640의 1/4 스케일)
disp_sad = compute_sad_cost_volume(gray_L, gray_R, max_disp=160, win=7)

epe_sad = compute_epe(disp_sad, gt_disp)
print(f"SAD Cost Volume EPE: {epe_sad:.2f} px")

# 비교용 SGBM도 같은 크기로
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=7,
    P1=8*3*7**2, P2=32*3*7**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disp_sgbm = sgbm.compute(img_L_small, img_R_small).astype(np.float32) / 16.0
epe_sgbm = compute_epe(disp_sgbm, gt_disp)
print(f"StereoSGBM EPE: {epe_sgbm:.2f} px")

# 시각화
vmax = gt_disp.max()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(gt_disp, cmap='plasma', vmin=0, vmax=vmax)
axes[0].set_title('GT Disparity (1/4 scale)')
axes[1].imshow(disp_sad, cmap='plasma', vmin=0, vmax=vmax)
axes[1].set_title(f'SAD Cost Volume (EPE={epe_sad:.1f}px)')
axes[2].imshow(disp_sgbm, cmap='plasma', vmin=0, vmax=vmax)
axes[2].set_title(f'StereoSGBM (EPE={epe_sgbm:.1f}px)')
plt.tight_layout()
plt.savefig('output_phase2_sad.png', dpi=100)
plt.show()
print("저장 완료: output_phase2_sad.png")