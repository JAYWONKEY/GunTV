import cv2
import numpy as np
import time
from ultralytics import YOLO
import pygame
import imageio
from PIL import Image, ImageSequence
import os

# 게임 변수
score = 0
health = 100
attack_count = 0
bomb_active = False  
bomb_timer = 0
bomb_effect_active = False
rifle_active = False
bomb_timer = 0
rifle_timer = 0
game_over = False
character_x = 200
mushroom_x = 900
mushroom_y = 500
mushroom_state = "stand"  # stand, move, hit, die
mushroom_frame_idx = 0
mushroom_hit_timer = 0
mushroom_hit_count = 0
mushroom_direction = 1  # 1: 오른쪽, -1: 왼쪽
mushroom_move_timer = time.time()
mushroom_move_interval = 3.0  # 3초마다 이동 방향 변경
umbrella_detected = False
bomb_detected = False
rifle_detected= False
# 이동 및 공격 변수
move_speed = 40  # 캐릭터 이동 속도 (2배 증가)
last_action_time = 0
action_cooldown = 0.1  # 액션 쿨다운 시간


# 모델 초기화 
model = YOLO("D:/python/gametest/testgame/img/best.pt")  # YOLO 모델 로드

# pygame 초기화 (소리 재생용)
pygame.mixer.init()
try:
    pygame.mixer.music.load('FloralLife.mp3')  # 배경음악
    pygame.mixer.music.play(-1)  # 무한 반복
except:
    print("일부 소리 파일을 로드할 수 없습니다.")

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 창 이름 설정
cv2.namedWindow("MapleStory Game", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MapleStory Game", 1280, 720)

# GIF 로드 함수
def load_gif(filepath):
    try:
        gif = imageio.mimread(filepath)
        frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        print(f"로드된 GIF 프레임 수: {len(frames)}, 파일: {filepath}")
        return frames
    except Exception as e:
        print(f"GIF 로드 오류 ({filepath}): {e}")
        return [np.ones((150, 150, 3), dtype=np.uint8) * 200]  # 기본 회색 프레임

# 디버깅: 이미지 폴더 내용 확인
img_folder = 'testgame\img'
if os.path.exists(img_folder):
    print(f"{img_folder} 폴더 내용:")
    for file in os.listdir(img_folder):
        print(f"  - {file}")
else:
    print(f"{img_folder} 폴더를 찾을 수 없습니다.")


# 배경 이미지 로드
try:
    background = cv2.imread('img/field.png', cv2.IMREAD_COLOR)
    background = cv2.resize(background, (1280, 720))
    print("배경 이미지 크기:", background.shape)
except Exception as e:
    print(f"배경 이미지 로드 오류: {e}")
    background = np.ones((720, 1280, 3), dtype=np.uint8) * 100  # 기본 배경

# 캐릭터 이미지 로드
try:
    character = cv2.imread('img/character.png', cv2.IMREAD_UNCHANGED)
    if character is None:
        print("캐릭터 이미지를 찾을 수 없습니다.")
        character = np.zeros((100, 50, 4), dtype=np.uint8)
        character[:, :, :3] = [0, 0, 255]  # 빨간색
        character[:, :, 3] = 255  # 불투명
    else:
        print("캐릭터 이미지 크기:", character.shape)
except Exception as e:
    print(f"캐릭터 이미지 로드 오류: {e}")
    character = np.zeros((100, 50, 4), dtype=np.uint8)
    character[:, :, :3] = [0, 0, 255]  # 빨간색
    character[:, :, 3] = 255  # 불투명
# 머쉬룸 GIF 로드
try:
    print("머쉬룸 GIF 로드 중...")
    mush_stand_frames = load_gif('img/mush_stand.gif')
    mush_move_frames = load_gif('img/mush_move.gif')
    mush_hit_frames = load_gif('img/mush_hit.gif')
    mush_die_frames = load_gif('img/mush_die.gif')
except Exception as e:
    print(f"머쉬룸 GIF 로드 오류: {e}")
    # 대체 이미지 사용
    print("대체 이미지 사용")
    
    # 단일 색상 프레임 생성 (각각 다른 색상으로 구분)
    dummy_frame_stand = np.ones((150, 150, 3), dtype=np.uint8) * np.array([50, 200, 50], dtype=np.uint8)  # 초록색
    dummy_frame_move = np.ones((150, 150, 3), dtype=np.uint8) * np.array([200, 50, 50], dtype=np.uint8)   # 빨간색
    dummy_frame_hit = np.ones((150, 150, 3), dtype=np.uint8) * np.array([50, 50, 200], dtype=np.uint8)    # 파란색
    dummy_frame_die = np.ones((150, 150, 3), dtype=np.uint8) * np.array([200, 200, 50], dtype=np.uint8)   # 노란색
    
    # 중앙에 텍스트 추가
    cv2.putText(dummy_frame_stand, "STAND", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(dummy_frame_move, "MOVE", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(dummy_frame_hit, "HIT", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(dummy_frame_die, "DIE", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 더미 프레임 시퀀스 생성
    mush_stand_frames = [dummy_frame_stand]
    mush_move_frames = [dummy_frame_move]
    mush_hit_frames = [dummy_frame_hit]
    mush_die_frames = [dummy_frame_die]

# 폭발 효과 이미지 로드
try:
    explosion_img = cv2.imread('img/bomb_effect.png', cv2.IMREAD_UNCHANGED)
    if explosion_img is None:
        print("폭발 효과 이미지를 찾을 수 없습니다.")
        # 간단한 원형 폭발 효과 생성
        explosion_img = np.zeros((100, 200, 4), dtype=np.uint8)
        # 빨간색 원형 그리기
        cv2.circle(explosion_img, (100, 50), 40, (0, 0, 255, 255), -1)
        # 노란색 작은 원형들 추가
        for i in range(8):
            angle = i * np.pi / 4
            x = int(100 + 50 * np.cos(angle))
            y = int(50 + 50 * np.sin(angle))
            cv2.circle(explosion_img, (x, y), 10, (0, 255, 255, 255), -1)
    else:
        explosion_img = cv2.resize(explosion_img, (200, 100))
        print("폭발 효과 이미지 크기:", explosion_img.shape)
except Exception as e:
    print(f"폭발 효과 이미지 로드 오류: {e}")
    explosion_img = None

# 투명 이미지 오버레이 함수
def overlay_transparent(background, overlay, x, y):
    """배경 이미지 위에 투명 이미지를 합성하는 함수"""
    # 입력 검사
    if overlay is None or background is None:
        return background
    
    # 채널 수 확인
    if len(overlay.shape) < 3 or overlay.shape[2] < 3:
        return background
    
    # 알파 채널 추가 (없는 경우)
    if overlay.shape[2] < 4:
        overlay = np.dstack([overlay, np.ones(overlay.shape[:2], dtype=np.uint8) * 255])
    
    h, w = overlay.shape[:2]
    
    # 경계 검사
    if y < 0 or y + h > background.shape[0] or x < 0 or x + w > background.shape[1]:
        # 이미지가 화면을 벗어나는 경우, 보이는 부분만 합성
        y_start = max(0, y)
        y_end = min(background.shape[0], y + h)
        x_start = max(0, x)
        x_end = min(background.shape[1], x + w)
        
        overlay_y_start = max(0, -y)
        overlay_y_end = h - max(0, (y + h) - background.shape[0])
        overlay_x_start = max(0, -x)
        overlay_x_end = w - max(0, (x + w) - background.shape[1])
        
        # 크기가 맞지 않으면 합성 중단
        if (y_end - y_start) <= 0 or (x_end - x_start) <= 0 or \
           (overlay_y_end - overlay_y_start) <= 0 or (overlay_x_end - overlay_x_start) <= 0:
            return background
        
        try:
            # 합성할 영역 추출
            overlay_crop = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
            alpha = overlay_crop[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            
            # 배경 추출
            bg_crop = background[y_start:y_end, x_start:x_end]
            
            # 크기 확인
            if bg_crop.shape[:2] != overlay_crop.shape[:2]:
                print(f"오버레이 크기 불일치: bg_crop {bg_crop.shape}, overlay_crop {overlay_crop.shape}")
                return background
            
            # 알파 블렌딩
            merged = bg_crop * (1 - alpha) + overlay_crop[:, :, :3] * alpha
            background[y_start:y_end, x_start:x_end] = merged
        except Exception as e:
            print(f"오버레이 오류 (경계 외): {e}")
    else:
        try:
            # 이미지가 화면 내에 있는 경우
            alpha = overlay[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            
            # 배경 영역 추출
            bg_region = background[y:y+h, x:x+w]
            
            # 크기 확인
            if bg_region.shape[:2] != overlay.shape[:2]:
                print(f"오버레이 크기 불일치: bg_region {bg_region.shape}, overlay {overlay.shape}")
                return background
            
            # 알파 블렌딩
            merged = bg_region * (1 - alpha) + overlay[:, :, :3] * alpha
            background[y:y+h, x:x+w] = merged
        except Exception as e:
            print(f"오버레이 오류 (경계 내): {e}")
    
    return background
# 투명 이미지 오버레이 함수
def overlay_transparent(background, overlay, x, y):
    """배경 이미지 위에 투명 이미지를 합성하는 함수"""
    if overlay is None or background is None:
        return background

    if len(overlay.shape) < 3 or overlay.shape[2] < 3:
        return background
    
    if overlay.shape[2] < 4:
        overlay = np.dstack([overlay, np.ones(overlay.shape[:2], dtype=np.uint8) * 255])

    h, w = overlay.shape[:2]

    if y < 0 or y + h > background.shape[0] or x < 0 or x + w > background.shape[1]:
        return background

    bg_region = background[y:y+h, x:x+w]
    overlay_region = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=-1)

    merged = bg_region * (1 - alpha) + overlay_region * alpha
    background[y:y+h, x:x+w] = merged
    return background

# 메인 게임 루프
print("게임 시작!")
while True:
    # 1. 웹캠 프레임 읽기
    ret_cam, cam_frame = cap.read()
    if not ret_cam:
        print("웹캠 프레임을 읽을 수 없습니다.")
        break
    
    # 좌우 반전 (거울 효과)
    cam_frame = cv2.flip(cam_frame, 1)
    
    # 2. 배경 이미지 복사
    game_frame = background.copy()
    
    # 3. YOLO로 객체 감지
    yolo_results = model(cam_frame, verbose=False)
    
    # 감지된 객체 처리
    for r in yolo_results:
        boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])  # 클래스 번호
        conf = float(box.conf[0])  # 신뢰도
        class_name = model.names[cls]  # 클래스 이름
        
        # Bounding box 좌표 추출
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # box.xyxy[0]은 [x1, y1, x2, y2] 좌표
        
        # 신뢰도가 일정 임계값 이상일 때만 처리
        if conf > 0.3:  # 신뢰도 임계값 설정
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

            # 폭탄 상태 확인 및 발사
    if bomb_detected and not bomb_active:
        bomb_active = True  # 폭탄 활성화
        bomb_timer = time.time()  # 폭탄 발사 시간 기록
        print(f"폭탄 발사! 타이머: {bomb_timer}")

    # 폭탄 상태 표시
    if bomb_active:
        # 현재 시간에서 폭탄 발사 후 경과된 시간 차이 계산
        current_time = time.time()  # 현재 시간
        elapsed_time = current_time - bomb_timer  # 폭탄 발사 후 경과 시간
        
        # 경과 시간 출력 (디버깅용)
        print(f"폭탄이 활성화된 지 {elapsed_time:.1f}초 경과됨.")
        
        if elapsed_time > 0.1:  # 0.1초 이상 경과 시 폭탄 비활성화
            bomb_active = False
            print("폭탄 비활성화!")
        else:
            print(f"폭탄 발사 중! 타이머: {bomb_timer}, 경과 시간: {elapsed_time:.2f}초")

            # 화면에 폭탄 상태 표시
            cv2.putText(game_frame, f"BOMB ACTIVE: {'YES' if bomb_active else 'NO'}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 객체 감지
        if class_name == "BOMB":
            bomb_detected = True
        elif class_name == "RIFLE":
            rifle_detected = True
        elif class_name == "UMBRELLA":  # 우산 감지 시
            print("우산 감지됨! 총 발사 준비!")
            rifle_detected = True  # 우산을 총으로 처리
        elif class_name == "PET_BOTTLE":  # 패트병 감지 시
            print("패트병 감지됨! 폭탄 준비!")
            bomb_detected = True  # 패트병을 폭탄으로 처리

        
    
    # 5. 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()
    
    # 일정 시간마다 머쉬맘 이동 (움직임 추가)
    if mushroom_state != "die" and current_time - mushroom_move_timer > mushroom_move_interval:
        mushroom_move_timer = current_time
        mushroom_direction *= -1  # 방향 전환
        mushroom_state = "move"  # 이동 애니메이션으로 변경
        print(f"머쉬맘 방향 변경: {'오른쪽' if mushroom_direction > 0 else '왼쪽'}")
    
        
     # 머쉬맘 이동 로직
    if mushroom_state == "move":
        # 실제 이동 (1.5초 동안)
        if current_time - mushroom_move_timer < 1.5:
            mushroom_x += mushroom_direction * 5  # 속도 조절
            # 화면 경계 확인
            if mushroom_x < 600:
                mushroom_x = 600
                mushroom_direction = 1
            elif mushroom_x > 1000:
                mushroom_x = 1000
                mushroom_direction = -1
        else:
            # 이동 후 다시 서 있는 상태로
            if mushroom_state != "hit" and mushroom_state != "die":
                mushroom_state = "stand"

     # 액션 쿨다운 확인
    if current_time - last_action_time > action_cooldown:
        # 키보드 입력으로 이동 및 공격
        if key == ord('a') or key == ord('A'):  # 왼쪽으로 이동
            character_x -= move_speed
            if character_x < 100:
                character_x = 100
            last_action_time = current_time
            print("왼쪽으로 이동")
            
        elif key == ord('d') or key == ord('D'):  # 오른쪽으로 이동
            character_x += move_speed
            if character_x > 800:
                character_x = 800
            last_action_time = current_time
            print("오른쪽으로 이동")
            
        elif key == 32:  # 스페이스바 (공격)
            if mushroom_state != "die":
                if bomb_detected and not bomb_active and not rifle_active:
                    bomb_active = True
                    bomb_timer = current_time
                    print("폭탄 공격!")
                    last_action_time = current_time
                    
                elif rifle_detected and not bomb_active and not rifle_active:
                    rifle_active = True
                    rifle_timer = current_time
                    print("라이플 공격!")
                    last_action_time = current_time

    # 6. 캐릭터 그리기
    if character is not None:
        resized_character = cv2.resize(character, (150, 150))
        game_frame = overlay_transparent(game_frame, resized_character, character_x, mushroom_y)
    # 7. 머쉬룸 상태 업데이트 및 그리기
    try:
        if mushroom_state == "stand":
            if len(mush_stand_frames) > 0:
                current_frame = mush_stand_frames[mushroom_frame_idx % len(mush_stand_frames)]
            else:
                print("머쉬맘 stand 프레임 없음")
                current_frame = np.ones((150, 150, 3), dtype=np.uint8) * 200
        elif mushroom_state == "move":
            if len(mush_move_frames) > 0:
                current_frame = mush_move_frames[mushroom_frame_idx % len(mush_move_frames)]
            else:
                print("머쉬맘 move 프레임 없음")
                current_frame = np.ones((150, 150, 3), dtype=np.uint8) * 150
        elif mushroom_state == "hit":
            if len(mush_hit_frames) > 0.5:
                current_frame = mush_hit_frames[mushroom_frame_idx % len(mush_hit_frames)]
            else:
                print("머쉬맘 hit 프레임 없음")
                current_frame = np.ones((150, 150, 3), dtype=np.uint8) * 100
            
            # hit 상태 타이머
            if current_time - mushroom_hit_timer > 0.5:  # 0.5초 후 stand로 돌아감
                mushroom_state = "stand"
                mushroom_frame_idx = 0
        elif mushroom_state == "die":
            if len(mush_die_frames) > 0:
                current_idx = min(mushroom_frame_idx, len(mush_die_frames)-1)
                current_frame = mush_die_frames[current_idx]
            else:
                print("머쉬맘 die 프레임 없음")
                current_frame = np.ones((150, 150, 3), dtype=np.uint8) * 50
            
            # 죽음 애니메이션이 끝나면 게임 종료
            if len(mush_die_frames) > 0 and mushroom_frame_idx >= len(mush_die_frames) - 1:
                game_over = True
        
        # 애니메이션 리사이징
        if current_frame.shape[0] != 150 or current_frame.shape[1] != 150:
            current_frame = cv2.resize(current_frame, (150, 150))
        
        # 머쉬맘 그리기
        game_frame[mushroom_y:mushroom_y+150, mushroom_x:mushroom_x+150] = current_frame
        
        # 현재 상태 표시 (디버깅용)
        cv2.putText(game_frame, f"State: {mushroom_state}", (mushroom_x, mushroom_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    except Exception as e:
        print(f"머쉬맘 그리기 오류: {e}")
        # 오류 발생 시 간단한 사각형으로 대체
        cv2.rectangle(game_frame, (mushroom_x, mushroom_y), 
                     (mushroom_x + 150, mushroom_y + 150), (0, 165, 255), -1)

    

    # # 7. 게임에서 발생할 수 있는 이벤트 처리 (폭탄 및 라이플)
    # if bomb_detected:
    #     # 폭탄을 사용하는 코드 구현 (폭발 효과 등)
    #     print("폭탄을 사용할 수 있습니다!")
    #     bomb_active = True
    #     bomb_timer = current_time
    #     bomb_detected = True  # 폭탄을 사용한 후 상태 초기화
    if bomb_detected and not bomb_active and not rifle_active:
        bomb_active = True  # 폭탄 활성화
        bomb_timer = time.time()  # 폭탄 발사 시간 기록
        print("폭탄 발사!")
        bomb_detected = False  # 감지 상태 초기화


    if rifle_detected:
        # 총을 발사하는 코드 구현 (총 공격 효과 등)
        print("라이플을 사용할 수 있습니다!")
        rifle_active = True
        rifle_timer = current_time
        rifle_detected = False  # 총을 사용한 후 상태 초기화
    
    # 8. 웹캠 영상 작은 창에 표시
    cam_resized = cv2.resize(cam_frame, (320, 240))
    game_frame[20:20+240, 940:940+320] = cam_resized
    
    # 9. UI 정보 표시
    cv2.putText(game_frame, f"Score: {score}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 감지된 객체 표시
    cv2.putText(game_frame, f"BOMB: {'YES' if bomb_detected else 'NO'}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(game_frame, f"RIFLE: {'YES' if rifle_detected else 'NO'}", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(game_frame, f"UMBRELLA: {'YES' if umbrella_detected else 'NO'}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
     # 조작 방법 설명
    cv2.putText(game_frame, "Controls:", (20, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(game_frame, "A key = Move Left", (20, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(game_frame, "D key = Move Right", (20, 190), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(game_frame, "SPACE = Attack (with BOMB/RIFLE)", (20, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)    
     # 체력바 그리기
    health_width = int((health / 100) * 200)
    cv2.rectangle(game_frame, (mushroom_x, mushroom_y - 50), 
                    (mushroom_x + 200, mushroom_y - 40), (0, 0, 255), 1)
    cv2.rectangle(game_frame, (mushroom_x, mushroom_y - 50), 
                    (mushroom_x + health_width, mushroom_y - 40), (0, 0, 255), -1)
    
    # 남은 피격 횟수 표시
    remaining_hits = max(0, 3 - mushroom_hit_count)
    cv2.putText(game_frame, f"남은 피격: {remaining_hits}", (mushroom_x, mushroom_y - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 디버깅 정보 표시 (개발용)
    debug_info = f"프레임: {mushroom_frame_idx}, 상태: {mushroom_state}"
    cv2.putText(game_frame, debug_info, (20, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    # 10. 게임 오버 처리
    if game_over:
        cv2.putText(game_frame, "YOU WIN!", (500, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(game_frame, f"Final Score: {score}", (500, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(game_frame, "Press 'R' to restart", (500, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 게임 재시작
        if key == ord('r') or key == ord('R'):
            game_over = False
            score = 0
            health = 100
            mushroom_hit_count = 0
            mushroom_state = "stand"
            mushroom_frame_idx = 0
            print("게임 재시작!")
    
    # 최종 게임 화면 표시
    cv2.imshow("MapleStory Game", game_frame)

    # ESC 키를 누르면 종료
    if key == 27:  # ESC 키
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("게임 종료!")
