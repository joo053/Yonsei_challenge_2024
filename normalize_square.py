import json
import cv2
import numpy as np
import os

image_name_list = range(1,21)

for image_name in image_name_list:
    
    image_name = str(image_name).zfill(2)
    
    # JSON 파일 경로
    json_file_path = f'./dataset/{image_name}.json'
    # 이미지 파일 경로
    image_file_path = f'./dataset/{image_name}.png'
    # 처리할 폴리곤의 label 번호
    label_numbers = [1, 2, 3]  # 원하는 label 번호로 설정

    # 이미지 파일 이름과 확장자를 분리
    image_name, _ = os.path.splitext(os.path.basename(image_file_path))
    print(json_file_path)
    # JSON 파일 로드
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        
    for number in label_numbers:
        # 지정된 label 번호의 폴리곤 좌표 찾기
        points = None
        for shape in json_data['shapes']:
            if shape['label'] == str(number):
                points = np.array(shape['points'], dtype='float32')
                break
            
        if points is None:
            continue
        
        # 좌측 위, 우측 위, 좌측 아래, 우측 아래로 정렬
        top_left = min(points, key=lambda p: p[0] + p[1])           # x + y가 가장 작은 점
        bottom_left = min(points, key=lambda p: p[0] - p[1])        # x - y가 가장 작은 점
        top_right = min(points, key=lambda p: -p[0] + p[1])         # -x + y가 가장 작은 점
        bottom_right = min(points, key=lambda p: -p[0] - p[1])      # -x - y가 가장 작은 점

        # 정렬된 좌표를 변환 행렬의 원본 소스로 사용
        pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        # 변환하고자 하는 512x512 정사각형의 네 꼭지점 좌표
        pts_dst = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype='float32')

        # 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # 원본 이미지 불러오기
        image = cv2.imread(image_file_path)

        # 원본 이미지를 변형하여 512x512 정사각형으로 변환
        output_image = cv2.warpPerspective(image, matrix, (512, 512))

        # 변환된 이미지 저장 (이미지 이름에 언더바와 label 번호 추가)
        output_image_path = f'{json_file_path}_/{number}.png'
        cv2.imwrite(output_image_path, output_image)
