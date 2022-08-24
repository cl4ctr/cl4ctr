# The source code of 《CL4CTR: A Contrastive Learning framework for CTR Prediction》

The most core code is class BasicCL4CTR in the [BasicLayer.py](./model/BasiclLayer.py), in which three SSL losses(L_cl,
L_ali and L_uni)
are computed to regularize feature representations.

CL4CTR is a model-agnostic and simple framework, which can be easily applied to existing CTR models, such as FM, DeepFM.

### Datasets:

All four datasets are publicly available.

- [Frappe](https://www.baltrunas.info/context-aware/frappe)
- [ML-tag](https://grouplens.org/datasets/movielens/)
- [ML-1m](https://grouplens.org/datasets/movielens/1m/)
- [SafeDriver](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)