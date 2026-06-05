import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img_L = cv2.imread('data/Jadeplant/im0.png')
img_R = cv2.imread('data/Jadeplant/im1.png')

print(f"Left image shape: {img_L.shape}")
print(f"Right image shape: {img_R.shape}")

# PFM 파일 읽기 함수 (GT Disparity용)
def read_pfm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline().decode().strip()
        dims = f.readline().decode().strip().split()
        W, H = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((H, W))
        if scale < 0:
            data = data[::-1]  # flip vertically
    return data

gt_disp = read_pfm('data/Jadeplant/disp0.pfm')
print(f"GT Disparity shape: {gt_disp.shape}")
print(f"GT Disparity range: {gt_disp.min():.1f} ~ {gt_disp.max():.1f}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB))
axes[0].set_title('Left Image')
axes[1].imshow(cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB))
axes[1].set_title('Right Image')
axes[2].imshow(gt_disp, cmap='plasma')
axes[2].set_title('GT Disparity Map')
plt.tight_layout()
plt.savefig('output_phase1_data.png', dpi=100)
plt.show()
print("저장 완료: output_phase1_data.png")