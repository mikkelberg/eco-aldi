Burnin:
- source-only (train+val on source data)

Training:
- target (unlabeled)
- source (labeled)

Validation:
- target + source


```
DATASETS:
  BATCH_CONTENTS:
  - labeled_strong
  - unlabeled_strong
  BATCH_RATIOS:
  - 1
  - 1
  PRECOMPUTED_PROPOSAL_TOPK_TEST: 1000
  PRECOMPUTED_PROPOSAL_TOPK_TRAIN: 2000
  PROPOSAL_FILES_TEST: []
  PROPOSAL_FILES_TRAIN: []
  TEST:
  - controlled-conditions_val
  - pitfall-cameras_val
  TRAIN:
  - controlled-conditions_train
  UNLABELED:
  - pitfall-cameras_train
```


Single GPU (freja, the most powerful one): `export CUDA_VISIBLE_DEVICES=0`
(in nvidia-smi, it says this one is GPU 2, but apparently it's actually 0 (so this command is not necessary, but I'm keeping it for clarity))



Adjustments:
Crashed with `FloatingPointError: Predicted boxes or scores contain Inf/NaN. Training has diverged.`.
Perhaps the early predictions are just garbage and cause exploding/vanishing gradients because they're just so off. There is already a high threshold on the teacher (0.8), which could help this otherwise. So we increased the warmup period (1000 iterations for now) (where the losses are not used):
```
SOLVER:
    WARMUP_ITERS: 1000 
```
And enabled gradient clipping, which helps with exploding/vanishing gradiets https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
```
SOLVER:
  CLIP_GRADIENTS:
    ENABLED: true
```



neg ratio 40
70 / 15 / 15 split
categories (11):
```
[{
            "id": 0,
            "name": "araneae"
        },
        {
            "id": 1,
            "name": "cantharidae"
        },
        {
            "id": 2,
            "name": "carabidae"
        },
        {
            "id": 3,
            "name": "coccinellidae"
        },
        {
            "id": 4,
            "name": "dermaptera"
        },
        {
            "id": 5,
            "name": "diptera-hymenoptera"
        },
        {
            "id": 6,
            "name": "isopoda"
        },
        {
            "id": 7,
            "name": "myriapoda"
        },
        {
            "id": 8,
            "name": "opiliones"
        },
        {
            "id": 9,
            "name": "staphylinidae"
        },
        {
            "id": 10,
            "name": "coleoptera (larvae)"
        }]

```