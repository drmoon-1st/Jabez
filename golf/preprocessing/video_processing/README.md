<code>
D:.
├─test
│  ├─false
│  │  ├─jpg
│  │  └─json
│  └─true
│      ├─jpg
│      └─json
└─tf
    ├─balanced_true
    │  ├─crop_keypoint
    │  ├─crop_video
    │  └─video
    ├─false
    │  ├─crop_keypoint
    │  ├─crop_video
    │  ├─json
    │  └─video
    └─true
        ├─jpg
        ├─json
        └─video
</code>
현재 코드의 작업은 모두
tf 폴더 안에서 true, false, balanced_true 폴더에 접근하는 방식으로 이루어져 있다
