데이터셋 파일은 AI 허브에서 다운 [AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=65)

파일 구조 일부 수정
golfDataset\스포츠 사람 동작 영상(골프)\Training\Public\male 아래에 각 스윙별 폴더가 아닌,
tf폴더 아래

<pre lang="markdown"> ``` ├─balanced_true │ └─skel_csv ├─false │ ├─cropped │ ├─jpg │ ├─json │ └─skel_csv └─true ├─cropped ├─jpg ├─json └─skel_csv ``` </pre>

해당 구조로 작성함


tf는 swing_1~10
test는 swing_11과 false는 tf 데이터에서 일부 가져옴
