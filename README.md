# Stereo Matching & Depth Estimation

두 카메라 이미지에서 깊이를 복원하는 Stereo Matching을 전통 알고리즘과 딥러닝으로 구현 및 정량 평가했습니다.

**Dataset:** Middlebury Stereo 2014 (Jadeplant)  
**Evaluation Metric:** EPE (End Point Error, px)

---

## Results

| Method | Resolution | EPE (px) ↓ |
|--------|-----------|------------|
| StereoBM (OpenCV) | Full (2632×1988) | 170.6 |
| StereoSGBM (OpenCV) | Full (2632×1988) | 90.2 |
| **SAD Cost Volume (직접 구현)** | 1/4 (658×497) | **15.0** |
| **RAFT-Stereo (Deep Learning)** | 1/2 (1316×996) | **4.16** |

<img width="1553" height="489" alt="output_final_comparison" src="https://github.com/user-attachments/assets/2b101339-1f57-4b2a-89cb-f250d22081c1" />

---

## Background

### Epipolar Geometry

두 카메라는 동일한 3D 점을 서로 다른 위치에서 촬영합니다. 왼쪽 이미지의 한 점에 대응하는 오른쪽 이미지의 점은 **Epipolar Line** 위에 존재하며, 이를 통해 2D 탐색을 1D로 줄일 수 있습니다.

두 카메라의 기하학적 관계는 **Fundamental Matrix F** (픽셀 좌표계)와 **Essential Matrix E** (정규화 좌표계)로 표현됩니다.

### Disparity & Depth

$$Z = \frac{f \cdot B}{d}$$

- `Z`: 깊이 (depth, mm)
- `f`: 초점거리 (focal length, px) — 본 실험: 7315.238 px
- `B`: 베이스라인 (baseline, mm) — 본 실험: 380.135 mm
- `d`: 시차 (disparity, px)

Disparity가 클수록 카메라에 가까운 물체입니다. (data set의 기존 calib.txt 사용)

### Rectification

두 카메라의 이미지 평면을 동일한 수평선으로 정렬하는 과정입니다. Rectification 후 대응점 탐색이 수평 방향으로만 이루어져 연산 효율이 크게 향상됩니다.

### Matching Cost

| Method | 수식 | 특징 |
|--------|------|------|
| SAD | `Σ\|IL - IR\|` | 빠름, 노이즈에 민감 |
| SSD | `Σ(IL - IR)²` | SAD보다 큰 오차에 민감 |
| NCC | 정규화 상관계수 | 밝기 변화에 강건 |

---

## Implementation

### Phase 1 — OpenCV 기반 구현

StereoBM과 StereoSGBM을 사용해 첫 Disparity Map을 생성했습니다.

```python
# StereoBM — 빠르지만 노이즈 많음
bm = cv2.StereoBM_create(numDisparities=640, blockSize=15)
disp_bm = bm.compute(gray_L, gray_R).astype(np.float32) / 16.0

# StereoSGBM — Semi-Global Matching, 더 정확
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=640, blockSize=7,
    P1=8*3*7**2, P2=32*3*7**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disp_sgbm = sgbm.compute(img_L, img_R).astype(np.float32) / 16.0
```

**실패 케이스:** 검은 배경(텍스처 없는 영역)과 나뭇잎 경계(반복 패턴)에서 Disparity Map이 크게 깨짐 → Phase 2

<img width="1800" height="500" alt="output_phase1_stereo" src="https://github.com/user-attachments/assets/1698942f-490a-4b3d-afde-cf93769eed1e" />

### Phase 2 — SAD Cost Volume 직접 구현

OpenCV 함수 없이 SAD Cost Volume을 NumPy로 구현했습니다.

```python
def compute_sad_cost_volume(img_l, img_r, max_disp=64, win=7):
    H, W = img_l.shape
    cost_vol = np.full((H, W, max_disp), np.inf)

    for d in range(max_disp):
        shifted = np.zeros_like(img_r)
        if d == 0:
            shifted = img_r.copy()
        else:
            shifted[:, d:] = img_r[:, :W-d]

        diff = np.abs(img_l.astype(np.float32) - shifted.astype(np.float32))
        cost_vol[:, :, d] = cv2.boxFilter(diff, -1, (win, win))

    # Winner-Takes-All
    disp_map = np.argmin(cost_vol, axis=2).astype(np.float32)
    return disp_map
```

1/4 해상도에서 SAD가 StereoSGBM보다 낮은 EPE(15.0 vs 23.1px)를 달성했습니다. 다운샘플링으로 텍스처가 smoothing되면서 SAD matching이 유리하게 작용한 결과이며, SGBM 파라미터가 풀해상도 기준으로 튜닝되어 있어 작은 이미지에서 불리했습니다.

<img width="1800" height="500" alt="output_phase2_sad" src="https://github.com/user-attachments/assets/2689e404-84f6-44f2-80af-d2dd2937a69b" />

### Phase 3 — RAFT-Stereo 추론

pretrained RAFT-Stereo (raftstereo-middlebury.pth)로 추론하여 EPE **4.16px** 달성.  
전통 알고리즘 대비 약 **40배** 낮은 오차입니다.

### Phase 3 — Disparity → Depth Map 변환

`calib.txt`의 카메라 파라미터를 이용해 Disparity를 실제 깊이(mm)로 변환했습니다.

$$Z = \frac{f \cdot B}{d + d_{offs}}$$

`doffs`(두 카메라의 principal point x 차이)를 보정항으로 추가해 정확한 depth를 계산합니다.

```python
focal_length = 7315.238  # px
baseline = 380.135       # mm
doffs = 809.195          # principal point x offset

depth_map = (focal_length * baseline) / (disp_map + doffs)
```

<img width="1339" height="501" alt="output_phase3_depth_map" src="https://github.com/user-attachments/assets/4929df59-f2ce-4b9c-9f1e-fdbba8c03f84" />


### Phase 3 — 3D Point Cloud 시각화

Depth Map과 카메라 내부 파라미터로 각 픽셀을 3D 공간으로 역투영(back-projection)했습니다.

$$X = \frac{(u - c_x) \cdot Z}{f}, \quad Y = \frac{(v - c_y) \cdot Z}{f}$$

Plotly로 인터랙티브 Point Cloud를 생성해 화분, 식물, 배경의 3D 구조를 확인했습니다.

<img width="1746" height="984" alt="output_phase3_pointcloud" src="https://github.com/user-attachments/assets/f074c924-62fd-450e-b154-2a839f583ac1" />

---

## Limitations

1. **정적 장면만 실험** — 동적 물체(사람, 차량)에서 Disparity 오류 미검증
2. **단일 데이터셋** — Middlebury Jadeplant 1개 장면으로만 평가, 일반화 성능 미확인

---

## Future Work

- **GAN/Diffusion 기반 Depth Completion** — 희소 Depth를 Dense로 완성
- **Knowledge Distillation으로 RAFT-Stereo 경량화** — 임베디드 환경 실시간 구동 탐구
- **Image Restoration 프로젝트와 결합** — 블러 영상에서의 Stereo Matching 성능 저하 분석

---

## Environment

```
Python 3.13
opencv-python 4.13
numpy 2.4
torch 2.11 (CUDA 12.8) — RAFT-Stereo 추론
```

## References

- RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching
- Middlebury Stereo Datasets
