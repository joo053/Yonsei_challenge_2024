# Yonsei Challenge 2024
## Strategy
1. 서버 접속되면 일단 SW와 데이터부터 파악
    - SW 정상 작동 여부
    - 데이터
        - 데이터가 진단 이전에 2번 촬영된 환자가 얼마나 있나?
        - 얼마나 imbalance한가?
        - X-ray가 T-spine부터 L-spine까지 잘 찍혀있는가?
        - 어느 vertebrum이 fracture인지 명확한가?
            - T-11, T-12, L-1만 남기고 나머지 탈락
        - clinical info가 얼마나 쓸만한가?
2. 3명이 각각 500장 (총 1500장) 정도에 T11-L1에 바운딩박스를 그림. (좁게, 약 4시간정도)
3. YOLO 학습
4. 전체에 inference하여 좌표 파악. Patch Extraction.
5. Training
    - Detection
        - Deep Learning
            - 패치 기반의 딥러닝 진행. (쉬운 난이도)
                - 패치를 조금 더 크게 morphological filter를 통해 늘려야 함.
            - 모델 아키텍처 수정
                - Densenet? Resnet? ViT?
                - imbalanced dataset? - Augmentation? Sampling?
                - clinical information과 ensemble
                    - 환자 데이터에 어떤것이 있는지 미리 확인
                - clinical info를 어디에 넣을까?
                    - 앞에? 뒤에? - 보통 어떻게 하지?
            - 반복 Validation
                - Sampling
                    - Cross-validation? Bootstrapping?
                    - Stability 측정
                - 여러 번의 학습을 통해 fine-tunning. 좁혀야 함.
                - 모델 확정되면 발표자료를 위한 시각화
        - Radiomics
            - 라디오믹스 진행 (중간 난이도)
            - Deep-learning과 함께 사용
                - Resnet-50 사용 (기존 논문)
                - 패치 크기는 약간 키워서
                - Resnet 마지막에서 2번째 layer (average pooling layer)를 취득하여 Lasso 또는 Lasso-Cox
                    - 목표: 20개, gamma 조절 필요, 선행논문 gamma = 0.0126
                - 목표: 20개
            - Radiomics
                - 패치 크기는 유지
                - feature extraction: pyradiomics
                - feature selection: ICC>=80%, Spearman CC > 90%, Lasso 또는 Lasso-Cox
                - Machine learning:
                    - LightGBM, Random Forest
                    - Imbalance problem: Class Weight, Focal Loss 사용
                - Radiomic Signature 계산
                    - Radiomic-Deep_Learning_Feature Signature = Sigma(Normalized feature * Lasso weight)
    - Prediction
    
6. Visualization
    - Deep Learning
        - Patch의 위치를 먼저 예측
        - 해당 영역을 중심으로 CAM
    - Radiomics
        - Nomogram
        - Hierarchical Clustering

## Labelme installation
- Download at https://github.com/wkentaro/labelme/releases

- Change below:
```
# Changed code
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
```

- Run cmd & command in the Labelme folder:
```
pip install -e .
```

## Labelme for_loop
```
for %i in (*.json) do labelme_json_to_dataset %i -o %i_
```

## Delta-radiomics
- If use delta-radiomics, need to co-register between first-and-last image.
- but, we use only rectangle, Only move 4 vertices of a rectangle to vertices of a square (512*512).
```
python normalize_square.py
```

## Hierarchichal Clustering

```
python hierarchical_clustering.py
```