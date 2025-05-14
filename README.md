# GunTV# 4조

- YOLO 기획
    
    → 처음엔 gun detect → hazard 유해요인 위해 객체탐지를 하려 했음
    
    → 직접 촬영까지 해서 라벨링 진행할거라 임의의 우산=rifle(총), 음료수=bomb(폭탄)으로 가정하여 촬영후 roboflow 내 labeling 진행 
    
- 준비과정
    - roboflow
        - [https://app.roboflow.com/yolopro-9psnd/my-first-project-zfwet/upload](https://app.roboflow.com/yolopro-9psnd/my-first-project-zfwet/upload)
    
    ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image.png)
    
    - 데이터 업로드 후 데이터 셋  순서 대로 진행한다.
        1. 팀원 할당된 raw_data
        2. Annotating 진행
        3. Dataset 완성
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%201.png)
        
    - 클래스 ( bomb/ bumb/ rifle)
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%202.png)
        
    - 완성된 데이터셋 다운
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%203.png)
        
    - yolov11 폴더다운
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%204.png)
        
    - data.yaml
        
        ```jsx
        폴더 경로 설정
        
        절대 경로 사용함
        path: D:\Projects\YoloGunPro\dataset
        train: D:\Projects\YoloGunPro\dataset\train\images
        val: D:\Projects\YoloGunPro\dataset\valid\images
        
        nc: 3
        names: ['bomb', 'bumb', 'rifle']
        
        로보플로우 삭제 실행 
        roboflow:
          workspace: yolopro-9psnd
          project: my-first-project-zfwet
          version: 1
          license: CC BY 4.0
          url: https://universe.roboflow.com/yolopro-9psnd/my-first-project-zfwet/dataset/1
        ```
        
    - [train.py](http://train.py)
        - 가상환경은 (p39_yolo_train)
        
        ```jsx
        from ultralytics import YOLO
        def main():
         
            # Load a model 전이학습 파일 모델 들고옴 
            model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
        
            # Train the model with MPS
            results = model.train(data="D:\Projects\YoloGunPro\dataset\data.yaml",
                                        epochs=100,
                                        imgsz=640,
                                        batch=16,
                                        device=0,
                                        pretrained = True,
                                        save = True,
                                        save_period=10,
                                        single_cls = True
                                        
                                        )
            
            
         
         
        if __name__ == '__main__':
            import multiprocessing
            multiprocessing.freeze_support()
            main()
        
        ```
        
- 1차_train(25/05/12/월)
    - 훈련
        
        ```jsx
   
 
    - 결과
        - f1 curve
        
        ![F1_curve.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/F1_curve.png)
        
        - 최적 F1-score: **0.85**에서 **confidence 0.591**일 때
            - 이 지점을 기준으로 **threshold 조정**을 고려할 필요 있음 → 각 팀원마다 0.5 → 0.3, 0.x 다르게 고려한 fine-tuning 진행
        - P_curve
            
            ![P_curve.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/P_curve.png)
            
        - PR_curve
            
            ![PR_curve.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/PR_curve.png)
            
            - 전체 mAP@0.5: **0.938**
            하지만 **recall이 낮은 지점에서 precision이 매우 높음** → 모델이 높은 확신을 가지는 경우에는 거의 정확히 예측함.
            한 클래스에 집중된 결과일 수 있어 **클래스별 mAP 확인이 필요**.
        - R_curve
            
            ![R_curve.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/R_curve.png)
            
            - Confidence가 낮을수록 Recall이 높고, Confidence가 높을수록 Recall이 급격히 감소.
            - 모델이 **높은 신뢰도에서만 탐지하는 경향** → 이건 예측 수를 줄이고 정확도에 집중하는 학습 경향으로 볼 수 있음.
        - result
            
            ![results.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/results.png)
            
            - 전반적으로 **train/val loss 모두 꾸준히 감소**
            - mAP@0.5 및 mAP@0.5~0.95는 지속적으로 증가
            - 다만, **val/cls_loss**의 초기 급등 및 폭발적인 스파이크는 **초기 클래스 인식 문제 또는 불균형에 의한 흔들림**으로 볼 수 있음
    - 해석(TROUBLE_SHOOTING)
        - yaml 파일 내 class는 3개이나, **결과셋이 class 0 : bomb | class 1 : bumb | class 2 : rifle** 클래스 1개로 인식됨 → 나쁜 결과
        - bomb 사진이 rifle 사진에 비해 3배 이상이 많아 추가 촬영
        - [YOLO11n.pt](http://YOLO11n.pt)
        - 결과 분석
            
            ### 1. **클래스 불균형 및 인식 문제**
            
            - 전체 결과에서 **실제로는 3개 클래스가 존재**하지만, **결과에서는 클래스가 하나(bomb)만 탐지됨**
            - 특히 `bomb` 클래스 이미지가 다른 클래스(rifle 등)보다 **3배 이상 많았기 때문에**, 모델이 bomb 클래스에 **과적합(overfitting)** 된 것으로 판단
            - 이를 보완하기 위해 **rifle 클래스 데이터를 추가로 촬영**하여 학습을 진행했습니다.
            - 추가촬영
                - 다양한 각도 및 조명 내 촬영
                
                ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%205.png)
                
                ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%206.png)
                
                ![20250512_1203210.jpg](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/20250512_1203210.jpg)
                
                ![20250512_163136.jpg](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/20250512_163136.jpg)
                
            
            ### 📝 결론
            
            - 현재 모델은 **bomb 클래스에 매우 최적화**되어 있고, **다른 클래스에 대한 인식 성능은 부족**.
            - **rifle 클래스 중심으로 데이터 증강 및 재학습 진행 중**.
            - 향후 성능 비교를 위해 **클래스별 mAP 값 확인 및 confusion matrix 시각화** 필요

---

- 2차_train, Fine Tuning(25/05/13/화)
    
    ---
    
    **HISTORY**
    
    1. 모델 변경
        
        n→ [yolo11](http://yolo11m.pt/)n.pt epoches 30 (지원)
        
        n→ [yolo8n.pt](http://yolo8n.pt) epoches 50 (대길)
        
        n → [yolo11n.pt](http://yolo11m.pt/)(팀장님), epoches 50
        
    2. 모델 데이터 추가
        - 다양한 각도 재촬영
            
            **bomb 222 → 600장** 
            
            **bumb 19 → 57장**
            
            **rifle 62 → 186장**
            
    
    ---
    
- 2차 TROUBLE SHOOTING & HOW TO SOLVE
    - bomb이 압도적으로 많아서 0으로 학습됨
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%207.png)
        
    
    수정
    
    1. rifle data 추가
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%208.png)
        
    2. model(”YOLO11n.pt”) + epoches 150, epoches 50, epoches 30등 진행  
        
        ```jsx
        from ultralytics import YOLO
        def main():
         
            # Load a model 전이학습 파일 모델 들고옴 
            model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
        
            # Train the model with MPS
            results = model.train(data=r"D:\Projects\YoloGunTV\dataset\data.yaml",
                                        epochs=150,
                                        imgsz=640,
                                        batch=16,
                                        device=0,
                                        pretrained = True,
                                        save = True,
                                        patience = 30
                                        )
         
        if __name__ == '__main__':
            import multiprocessing
            multiprocessing.freeze_support()
            main()
        
        ```
        
    3. model(”YOLO8n.pt”) + epoches 150, 100, 50번 진행
        
        ```jsx
        results = model.train(
                data=r"C:/Users/702-09/Desktop/yolo_data/data.yaml",
                epochs=150,
                imgsz=640,
                batch=16,
                device=0,
                pretrained=True,
                save=True,
                lr0=0.001,             # 초기 학습률
                lrf=0.01,              # 최종 학습률 비율 (Cosine scheduler)
                weight_decay=0.0005,   # 가중치 감소 (정규화)
                warmup_epochs=3,# 워밍업 에폭 수
            )
        
        ```
        
        ---
        
        프레임마다 카메라 고정
        
        1fr 마다 객체가 바뀜 → 객체 감지  변화를 고정시킨다.
        
        ---
        
        5.13일 TroubleShooting
        
        rifle 사진 800장 튀기고 라벨링 진행 하였으나 여전히 detect시 못잡는 문제 발생
        
        ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%209.png)
        
        - detect시 result 내 *imgz*=**640**, *con*=**0.1 추가, con= 0.1~0.4 사이 train 진행**
        
        ---
        
        - 
        - 크롤링
            - 우산 이미지 크롤링
                - 접힌 우산등 150장 → dataset에 올린 뒤 이미지 라벨링 진행
                
                ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2010.png)
                
                - 이미지 크롤링 저장소
                
                ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2011.png)
                
                - 팀원별 라벨링 이미지 할당
                    
                    ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2012.png)
                    
                - 라벨링
                    
                    ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2013.png)
                    
                - 완성된 데이터셋
                    
                    ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2014.png)
                    
                
        - 전처리 과정
            
            ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2015.png)
            
            ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2016.png)
            
            ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2017.png)
            
            ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2018.png)
            
        - 환경설정
            - data.yaml
            
            ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2019.png)
            
        - 훈련
            - train.py
                
                ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2020.png)
                

---

- 3차_train (여러 케이스) (25/05/14/화)
    - gpu 이슈
        - 다양한 train Fine_Tuning → gpu 터짐
            
            ```jsx
                results = model.train(
                        data=r"D:\Projects\YOLO_RESULT\dataset\data.yaml",
                        epochs=30,
                        imgsz=640,
                        batch=16,
                        device=0,
                        pretrained=True,
                        save=True,
                        lr0=0.001,             # 초기 학습률
                        lrf=0.01,              # 최종 학습률 비율 (Cosine scheduler)
                        weight_decay=0.0005,   # 가중치 감소 (정규화)
                        warmup_epochs=3,      # 워밍업 에폭 수
                        patience = 30,
                        translate=0.2,
                        hsv_v=0.2,
                        hsv_h=0.01,
                        scale=0.5,      # 객체 스케일 변화
                        flipud=0.1,     # 수직 뒤집기
                        fliplr=0.5,     # 수평 뒤집기
            
                    )
            
            ```
            
    - 인터넷이슈
        - 인터넷 끊긴 경우 훈련이 안돌아가는 문제
        - 데이터가 많은 경우(2000개) epocs가 오래 걸림 (30분) → modelv8n
    - 여러 증강 기법으로 train
        
        모델(8n, 11n, 11s, 11m) train시 epochs= 50~200회, resume 추가
        
- 최종 result (팀장님)
    
    ![results.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/results%201.png)
    

---

- UI (25/05/14/화)
    - 메이플 스토리 머쉬맘 보스 몹 잡는 영상
    - 터미널에 객체 탐지로그는 있으나 이미지는 안나오는 이슈
        - TROUBLE SHOOTING
            
            ![image.png](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/image%2021.png)
            
            ```jsx
              # 신뢰도가 일정 임계값 이상일 때만 처리
                    if conf > 0.4:  # 신뢰도 임계값 설정
                        if class_name == "bomb":  # "bomb" 객체가 감지되면
                            bomb_detected = True  # 폭탄 감지 상태 업데이트
                        elif class_name == "PET_BOTTLE":  # 패트병 감지 시
                            print("패트병 감지됨! 폭탄 준비!")
                            bomb_detected = True  # 패트병을 폭탄으로 처리
                            # 폭발 효과 이미지 불러오기 및 화면에 표시
                            if explosion_img is not None:
                                # RGB에서 BGR로 변환
                                explosion_img = cv2.cvtColor(explosion_img, cv2.COLOR_RGB2BGR)
                                # 알파 채널이 없다면 추가
                                if explosion_img.shape[2] < 4:
                                    alpha_channel = np.ones((explosion_img.shape[0], explosion_img.shape[1]), dtype=np.uint8) * 255  # 불투명 알파
                                    explosion_img = cv2.merge([explosion_img, alpha_channel])  # 알파 채널 추가
                            # 감지된 객체에 바운딩 박스 그리기
                            cv2.rectangle(cam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 객체에 바운딩 박스 그리기
                            cv2.putText(cam_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 클래스명 및 신뢰도 표시
            
            ```
            

---

[bandicam 2025-05-14 17-46-23-261.mp4](4%E1%84%8C%E1%85%A9%201f170c722f2a80528a85ccb420be4d55/bandicam_2025-05-14_17-46-23-261.mp4)

 

---

- 발표 내용
- 피드백
