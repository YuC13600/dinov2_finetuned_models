# DINOv2 finetune細節

## 1. 模型架構

```
Image Input (224×224)
    ↓
DINOv2 Backbone (ViT-B/14)
    ↓ (768)
Feature Processor
    ├─ BatchNorm1d(768)
    └─ Dropout(0.2)
    ↓
Projection Head
    ├─ Linear(768 → 1024)
    ├─ ReLU
    ├─ Dropout(0.3)
    ├─ Linear(1024 → 1280)
    └─ BatchNorm1d(1280)
    ↓
L2 Normalization
    ↓
Embedding Output (1280)
```

### [DINOv2](https://github.com/facebookresearch/dinov2) Backbone
預訓練的Vision Transformer ViT-B/14。

### Feature Processor
利用BatchNorm1d穩定特徵分佈並用Dropout防止過擬合。

### Projection Head
利用Linear與ReLU學習線性與非線性特徵，並使用BatchNorm1d和Dropout穩定特徵並防止過擬合。

### L2 Normalization
將向量正規化到單位球面。

## 2. 訓練策略

### 階段 1: 只訓練Projection Head和Feature Processor (20 Epochs)
- 凍結Backbone
- Learning Rate: $3 \times 10^{-4}$

### 階段 2: 解凍Backbone最後2個Block (15 Epochs)
- Learning Rate: $8 \times 10^{-5}$

### 階段 3: 解凍Backbone最後4個Block (12 Epochs)
- Learning Rate: $3 \times 10^{-5}$

## 3. [Triplet Loss](https://en.wikipedia.org/wiki/Triplet_loss)

![triplet figure](./Triplet_Loss_Minimization.png)

$$ L(A, P, N) = Max(\parallel f(A) - f(P) \parallel ^ 2 - \parallel f(A) - f(N) \parallel ^ 2 + \alpha, 0) $$

公式推導:  
AP距離愈近愈好，AN距離愈遠愈好，所以  
$$ \parallel f(A) - f(P) \parallel ^ 2 \ \le \ \parallel f(A) - f(N) \parallel ^ 2 $$  
$$ \parallel f(A) - f(P) \parallel ^ 2 \ - \ \parallel f(A) - f(N) \parallel ^ 2 \le 0 $$  
但這樣有可能會讓所有輸出的特徵都為零，或是距離相減都為零，所以通常會加一個間隔$\alpha$(Margin)  
$$ \parallel f(A) - f(P) \parallel ^ 2 \ - \ \parallel f(A) - f(N) \parallel ^ 2 \le - \alpha $$  
$$ \parallel f(A) - f(P) \parallel ^ 2 \ - \ \parallel f(A) - f(N) \parallel ^ 2 + \alpha \le 0 $$  

**參數:**

$Margin(\alpha) = 0.3$

Mining Strategy: **Hard** ($AP距離-AN距離+\alpha > 0$)

> Triplet的三種類型:  
> 1. Easy Triples: AN距離大於AP距離，且距離差距大於Margin，A與P本來就夠近，也離N夠遠。  
> 2. Hard Triples: AP距離大於AN距離。  
> 3. Semi-hard Triplets: AN距離大於AP距離，但距離差距小於Margin，因此上面的公式還不成立。  

![Triplet Mining](./Triplet_mining.png)

## 4. 資料集分割

| Data Set | Proportion |
| --- | --- |
| Training Set | 70% |
| Validation Set | 15% |
| Test Set | 15% |

## 5. 優化

### 5.1 Mini-batch和Accumulation Steps

每個mini-batch大小16，累積8個mini-batch更新一次權重。

因此等效的batch size是128。

**參數:**

$batch\ size = 16$

$accumulation\ steps = 8$

### 5.2 Early Stopping

連續多個Epoch，Validation Loss沒有進步就直接結束目前階段。

**參數:**

$patience = 6$

$delta = 0.0005$

### 5.3 Gradient Clipping

利用Clip by Norm避免梯度爆炸。

**參數:**

$max\ norm = 1.0$

### 5.4 ReduceLROnPlateau

連續3個Epoch的Val Loss沒有改善的話就把Learning Rate減半。

**參數:**

$factor = 0.5$

$patience = 3$

$min\ lr = 10^{-6}$

### 5.5 Weight Decay

保持較小的權重避免overfitting。

標準梯度下降:  
$$ w_{new} = w_{old} - learning\ rate \times gradient $$  
加上 weight decay:  
$$ w_{new} = w_{old} - learning\ rate \times gradient - learning\ rate \times \lambda \times w_{old} $$  
可以整理為  
$$ w_{new} = (1 - learning\ rate \times \lambda) \times w_{old} - learning\ rate \times gradient $$  

**參數:**

$\lambda = 10^{-4}$

## 6. 變因與測試結果

| Model Name | Batch Size | Same Area Negative | Box Size | TestSet Accuracy | TestSet Loss | N-Benchmark Avg | N-Benchmark A37 | N-Benchmark A38 | N-Benchmark A39 | N-Benchmark A40 | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-trained Weight | <hr> | <hr> | bbox | <hr> | <hr> | 29.48% | 28.12% | 35.48% | 29.63% | 24.32% | <hr> |
| 20250812_152526 | 32 | ❌ | bbox | 92.6% | 0.1659 | 48.25% | 50.00% | 51.61% | 48.15% | 43.24% | <hr> |
| 20251007_133126 | 32 | ✅ | bbox | 88.8% | 0.2523 | 39.32% | 46.88% | 41.94% | 33.33% | 35.14% | <hr> |
| 20251008_094017 | 16 | ✅ | bbox | 90.4% | 0.1636 | 40.19% | 37.50% | 48.39% | 37.04% | 37.84% | <hr> |
| 20251008_234015 | 64 | ✅ | bbox | N/A | N/A | N/A | N/A | N/A | N/A | N/A | CUDA out of memory |
| 20251014_183603 | 16 | ❌ | bbox | 92.8% | 0.1012 | 40.97% | 37.50% | 38.71% | 44.44% | 43.24% | <hr> |
| Pre-trained Weight | <hr> | <hr> | whole image | <hr> | <hr> | 50.88% | 34.38% | 54.84% | 62.96% | 51.35% | <hr> |
| **20251015_165008 (Final Model)** | 16 | ✅ | whole image | 92.7% | 0.1330 | 64.43% | 62.50% | 61.29% | 55.56% | 78.38% | <hr> |
| 20251016_133229 | 16 | ❌ | whole image | 97.9% | 0.0429 | 63.31% | 56.25% | 58.06% | 74.07% | 64.86% | <hr> |

> `Same Area Negative`: 只使用與Anchor, Positive同一個區域內的其他珊瑚作為Negative。  
> `N-Benchmark(Nearest Benchmark)`: 2022和2023間Tag 37-40各區域的配對正確率。

## Reference

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

[DINOv2 GitHub Repository](https://github.com/facebookresearch/dinov2)