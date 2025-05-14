# 🔫 GunTV 프로젝트 (4조)

> YOLO를 활용한 총기 및 폭발물 의심 객체 인식 + 메이플스토리 스타일 UI 연동 프로젝트

---

## 📌 프로젝트 개요

- YOLO 기반 객체 탐지 모델을 활용해 **실생활 위험 요소(총기, 폭탄 등)**를 탐지하는 프로젝트입니다.
- 실제 촬영 데이터를 사용하여 우산 = 총, 음료수병 = 폭탄으로 가정하고 훈련합니다.
- 메이플스토리 UI 스타일로 감지 정보를 시각화하여 게임 형식의 위험 감지 시스템을 구현합니다.

---

## 🧪 준비 과정

### 1. 촬영 및 라벨링

- 총 = 우산, 폭탄 = 음료수로 대체해 직접 촬영
- Roboflow에서 라벨링 작업
- 라벨 클래스: `bomb`, `bumb`, `rifle`

**👉 [Roboflow 링크](https://app.roboflow.com/yolopro-9psnd/my-first-project-zfwet/upload)**

### 2. 데이터셋 구성

- 클래스별 이미지 수 불균형 → `rifle` 클래스 추가 촬영
- 최종 데이터셋 구성:
    - bomb: 600장
    - bumb: 57장
    - rifle: 186장 → 이후 800장 추가 촬영

### 3. `data.yaml` 설정 

```yaml
path: D:\Projects\YoloGunPro\dataset
train: D:\Projects\YoloGunPro\dataset\train\images
val: D:\Projects\YoloGunPro\dataset\valid\images

nc: 3
names: ['bomb', 'bumb', 'rifle']
🏋️ 모델 학습
사용 모델
YOLOv8n

YOLOv11n, 11s, 11m 등 다양한 경량/중간 모델

```

```
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # 전이학습 모델 사용
    results = model.train(
        data=r"D:\Projects\YoloGunTV\dataset\data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        pretrained=True,
        save=True,
        patience=30,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        translate=0.2,
        hsv_v=0.2,
        hsv_h=0.01,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
```

📊 훈련 결과 및 분석
성능 평가
F1-score 최고점: 0.85 (conf=0.591)

mAP@0.5: 0.938

클래스 불균형 문제: bomb 클래스 과적합

주요 그래프
F1 / Precision / Recall / PR 커브 확인

클래스별 mAP 및 confusion matrix 필요

문제점 & 해결
문제	해결
bomb 과적합	rifle 이미지 다량 추가
클래스 1개만 감지	클래스 균형 조정
감지 시 이미지 표시 안됨	UI 코드 수정으로 해결
모델 성능 불안정	다양한 모델/에폭 실험 및 증강 기법 적용

🎮 UI 개발
목표
메이플스토리 스타일 보스 전투 인터페이스 구현

카메라에서 객체 감지 시 공격 애니메이션 표시

UI 트러블슈팅
이미지 안나오는 문제 → conf > 0.4 조건 아래에서만 이미지 표시되도록 코드 수정
```
if conf > 0.4:
    if class_name == "bomb":
        bomb_detected = True
    elif class_name == "PET_BOTTLE":
        print("패트병 감지됨! 폭탄 준비!")
        bomb_detected = True
        if explosion_img is not None:
            explosion_img = cv2.cvtColor(explosion_img, cv2.COLOR_RGB2BGR)
            if explosion_img.shape[2] < 4:
                alpha = np.ones((explosion_img.shape[0], explosion_img.shape[1]), dtype=np.uint8) * 255
                explosion_img = cv2.merge([explosion_img, alpha])
        cv2.rectangle(cam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cam_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```
🧠 향후 계획
클래스별 mAP, confusion matrix 정밀 분석

실시간 UI 안정성 개선 (카메라 고정 등)

다양한 실생활 객체(패트병, 장난감 총 등) 추가 라벨링

YOLOv8/11 최종 모델 비교 후 선택


👥 팀원
4조 YOLO조
설태수 팀장님
홍대길 팀원
박지원 팀원

📷 시연 이미지 (일부)
원본 데이터 라벨링	결과 감지 화면

📌 비고
로컬 실행 경로: D:\Projects\YoloGunPro\

모델: YOLOv11n.pt, YOLOv8n.pt 등 혼합 사용

개발 환경: Python 3.9, Ultralytics YOLO, OpenCV, Roboflow

- 노션내 자세한 내용으로 볼 수 있습니다.
- https://sixth-mink-52f.notion.site/4-1f170c722f2a80528a85ccb420be4d55?pvs=73
