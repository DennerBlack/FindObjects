# Нахождение маркеров на изображении
## Описание
Задача нахождения объектов на изображении - классическая задача компьютерного зрения. Но в данном наборе изобрачений искомые объекты, маркеры, как и сами изображенмя имеют черезвычайно высокий уровень шумов, из-за того что это скриншоты облака точек.
## Стркутура проекта 
```
.
├── data
│   ├── dataset
│   │   ├── hard
│   │   ├── normal
│   │   └── test
│   └── images
│       ├── Screenshot_4.png
│       ├── target
│       ├── target_source.png
│       ├── template.png
│       ├── tmp2.png
│       └── validation
├── main.py
├── Prepare.py
├── requirements.txt
├── SCNN.py
├── SCNN_train.py
└── weights
    ├── scnn_E100_v0_hard.h5
   ...
    └── scnn_E50_v1_normal.h5
```
