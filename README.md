# Golf Swing AI analysis
골프 스윙 영상을 입력으로 받아,  
Timesformer, stgcn을 통한 임베딩 추출후  
이를 결합하여 MLP에 훈련하는 이진 분류 모델  

This is binary classification model that takes a golf swing video as input, 
extracts embeddings using Timesformer and stgcn,  
and combines them to train an MLP.

```
┌───────────────────────┐       ┌──────────────────────┐
│     RGB Frames        │       │  Skeleton Sequence   │
│   (T, H, W, 3)        │       │     (T, J, C)        │
└─────────┬─────────────┘       └─────────┬────────────┘
          │                               │
          ▼                               ▼
┌───────────────────────┐       ┌──────────────────────┐
│     TimeSformer       │       │    ST-GCN(mmAction2) │
│  (Patch → Embedding   │       │  (Spatial-Temporal   │
│    → Transformer)     │       │   Graph Conv Layers) │
└─────────┬─────────────┘       └─────────┬────────────┘
          │                               │
          ▼                               ▼
┌───────────────────────┐       ┌──────────────────────┐
│  [CLS] Embedding      │       │   Skeleton Embedding │
│    (D-dim vector)     │       │    (256-dim vector)  │
└─────────┬─────────────┘       └─────────┬────────────┘
          │               Feature Fusion (Concat / Attention)
          └───────────────────────────────┬
                                          │
                                          ▼
                              ┌───────────────────────────┐
                              │    Classifier             │
                              │   (FC → Softmax)          │
                              └────────────┬──────────────┘
                                           │
                                           ▼
                              ┌───────────────────────────┐
                              │   Output: Good / Bad      │
                              └───────────────────────────┘
```

# Environment Setting
도커로 제공 준비중...  
Under development in Docker...  

  
# Code Structure  
### Data Preprocessing
1. aihub에서 데이터를 다운로드
2. action_classfication 코드를 통해, male, female 폴더 순회하며 새롭게 dataset 디렉토리를 만들고 true, false로 분리함
    - 분리 과정은 json 파일을 분석하여 evalutions 결과를 바탕으로 분류, dataset 아래에 true/json, true/jpg, false/json, false/jpg에 저장한다
    - 저장시 golfdataset/dataset 로 이동해버림
3. jpgs_to_mp4로 1곳에 모인 모든 jpg들 mp4로 변환(기존 jpg는 삭제)
    - 예를 들어, 100개의 영상이 생성되었을 때, jpg를 삭제한다. 100개단위로 savepoint를 설정하고, 오류가 나더라도, 데이터가 손실되지 않도록 한다.
4. balanced_true_extracted를 통해 false 와 1대1 비율로 true 데이터를 추출한다
5. video_crop_and_csv_merged를 통해 openpose를 통한 video crop과 csv 추출을 동시에 진행한다
#
1. AI Hub Data Download
2. Dataset Creation via Action Classification
    - Use the action_classification code to iterate through male and female folders and create a new dataset directory. Data is separated into true and false by analyzing JSON files based on evaluations results. The classified files are saved to golfdataset/dataset/true/json, golfdataset/dataset/true/jpg, golfdataset/dataset/false/json, and golfdataset/dataset/false/jpg.
3. Use jpgs_to_mp4 to convert all JPG files in a single location to MP4 format. The original JPGs are deleted after conversion. A savepoint is set every 100 videos to prevent data loss in case of errors.
4. Use balanced_true_extracted to extract true data at a 1:1 ratio with the false data.
5. Use video_crop_and_csv_merged to simultaneously perform video cropping with OpenPose and extract CSV files.

# Traing, Test
1. golf/fusion의 data_loader.ipynb에서 timesformer, stgcn의 사전학습된 혹은 파인튜닝된 모델을 파라미터에 입력  
    - 백본까지 학습한 파인튜닝을 하기 싫다면, timesformer, stgcn에서 제공하는 사전학습 모델을 사용하면 됨  
2. data_loader를 통해 임베딩.npy들 추출  
3. main_classfier를 통해 MLP 학습 & 테스트 진행  
#  
1. Inputting Pre-trained Models  
In golf/fusion/data_loader.ipynb, input either pre-trained or fine-tuned models for Timesformer and STGCN into the parameters.  
          If you do not want to perform fine-tuning that includes backbone training, you can use the pre-trained models provided by Timesformer and STGCN.  

2. Extracting Embeddings  
Extract the embedding .npy files using the data_loader.  

3. Training & Testing the Classifier  
Perform MLP training and testing using the main_classifier.  
---
### Data

데이터셋 파일은 AI 허브에서 다운 [AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=65)

파일 구조 일부 수정
golfDataset\스포츠 사람 동작 영상(골프)\Training\Public\male 아래에 각 스윙별 폴더가 아닌,
tf폴더 아래

```
├─balanced_true
│  └─skel_csv
├─false
│  ├─cropped
│  ├─jpg
│  ├─json
│  └─skel_csv
└─true
   ├─cropped
   ├─jpg
   ├─json
   └─skel_csv
```


해당 구조로 작성함


tf는 swing_1~10
test는 swing_11과 false는 tf 데이터에서 일부 가져옴
