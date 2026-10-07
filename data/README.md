# Data

Datasets are intentionally excluded from version control. Later phases will add deterministic COCO
subset generation under `data/subsets/`. Expected COCO layout:

```text
data/coco/
├── annotations/person_keypoints_train2017.json
├── annotations/person_keypoints_val2017.json
├── train2017/
└── val2017/
```

Download COCO only from its official distribution. No dataset or video is bundled with this project.
