from ultralytics import YOLO
import cv2 # OpenCV 추가 (cv2.imshow 등을 사용하지 않더라도 r.plot() 내부에서 사용될 수 있음)

# 모델 로드
model = YOLO(r"D:/Projects/TEST/best.pt")

# 웹캠에서 실시간으로 추적 수행
# source=0은 기본 웹캠을 의미합니다.
# show=True를 사용하면 감지 및 추적 결과가 포함된 영상이 화면에 실시간으로 표시됩니다.
# stream=True는 비디오 스트림을 효율적으로 처리하기 위해 필요합니다.
# tracker="botsort.yaml"은 BoT-SORT 추적기를 사용하도록 지정합니다.
# persist=True는 추적 ID를 프레임 간에 유지하려고 시도합니다.
print("웹캠을 시작합니다. 종료하려면 결과 창에서 'q' 키를 누르거나 Ctrl+C를 누르세요.")
results = model.track(source=0, show=True, stream=True, persist=True, tracker="botsort.yaml", imgsz=640) # imgsz=640 추가

# 현재 사용 중인 conf 값 확인 및 출력
if hasattr(model, 'predictor') and model.predictor is not None and hasattr(model.predictor, 'args'):
    print(f"현재 사용 중인 신뢰도 임계값 (conf): {model.predictor.args.conf}")
else:
    print("모델 predictor 또는 내부 인자(args)를 찾을 수 없어 conf 값을 표시할 수 없습니다.")
    print("model.track() 또는 model.predict()가 한 번 이상 호출된 후에 시도해 보세요.")

# 각 프레임 결과 처리
for r in results:
    # r.plot()은 감지/추적 결과가 그려진 프레임을 반환합니다.
    # show=True로 인해 이 프레임은 자동으로 화면에 표시됩니다.
    # annotated_frame = r.plot() # 필요시 이 프레임을 받아 추가 작업 가능

    # 감지된 객체가 있고, 추적 ID가 있는 경우 정보 출력
    if r.boxes is not None and hasattr(r.boxes, 'id') and r.boxes.id is not None:
        print(f"--- 프레임 ---")
        tracked_boxes = r.boxes.id.int().tolist() # 추적 ID 리스트
        
        for i, track_id in enumerate(tracked_boxes):
            xyxy = r.boxes.xyxy[i].tolist()     # 바운딩 박스 좌표
            conf_score = r.boxes.conf[i].item()       # 신뢰도 점수
            cls_id = int(r.boxes.cls[i].item()) # 클래스 ID
            class_name = r.names[cls_id]      # 클래스 이름

            print(f"  추적 ID: {track_id}, 클래스: {class_name}, 신뢰도: {conf_score:.2f}, 박스: [{', '.join(f'{c:.0f}' for c in xyxy)}]")
    
    # 만약 show=True를 사용하지 않고 직접 화면에 표시하려면:
    # if annotated_frame is not None:
    #     cv2.imshow("YOLOv8 Tracking", annotated_frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

print("웹캠 테스트가 종료되었습니다.")
# 만약 cv2.imshow를 사용했다면 창 닫기
# cv2.destroyAllWindows()