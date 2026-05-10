# ??/?? LADP ??????

| ?? | ?? | ??LEO?? | ??LEO | ????? | ??? |
|---|---|---:|---|---:|---|
| homogeneous_5_5_5_tflops | YOLOv5 | 1 | SAT-01 | 0.373 | `{"SAT-01": [0, 3], "GS": [4, 23]}` |
| homogeneous_5_5_5_tflops | VGG19 | 3 | SAT-01 -> SAT-02 -> SAT-03 | 0.643 | `{"SAT-01": [0, 4], "SAT-02": [5, 18], "SAT-03": [19, 39], "GS": [40, 44]}` |
| homogeneous_5_5_5_tflops | Swin-Base | 1 | SAT-01 | 0.990 | `{"SAT-01": [0, 4], "GS": [5, 5]}` |
| homogeneous_5_5_5_tflops | ViT-Huge | 0 |  | 1.000 | `{"GS": [0, 32]}` |
| heterogeneous_3_8_5_tflops | YOLOv5 | 1 | SAT-01 | 0.378 | `{"SAT-01": [0, 3], "GS": [4, 23]}` |
| heterogeneous_3_8_5_tflops | VGG19 | 3 | SAT-01 -> SAT-02 -> SAT-03 | 0.645 | `{"SAT-01": [0, 4], "SAT-02": [5, 18], "SAT-03": [19, 39], "GS": [40, 44]}` |
| heterogeneous_3_8_5_tflops | Swin-Base | 1 | SAT-02 | 0.788 | `{"SAT-02": [0, 4], "GS": [5, 5]}` |
| heterogeneous_3_8_5_tflops | ViT-Huge | 0 |  | 1.000 | `{"GS": [0, 32]}` |