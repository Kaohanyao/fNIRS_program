# fNIRS 深度学习平台基础知识

本平台面向 fNIRS 信号处理、深度学习建模、subject-wise 验证、可解释性分析和 RAG 知识库管理。

## 数据处理
fNIRS 数据常见格式包括 SNIRF、NIRS、MAT 和 CSV。实验前需要确认采样率、通道名称、事件标记、标签分布和被试编号。

## 预处理
常见流程包括光密度转换、Beer-Lambert 转换、TDDR 运动伪影校正、带通滤波、基线校正和事件锁定 epoch 提取。

## 建模与验证
fNIRS 深度学习可使用 fNIRS-EEGNet、CNN-LSTM、TCN、Graph-TCN 和 Hybrid 3D CNN。验证应优先采用 LOSO 或 Group K-Fold，避免 trial-level 随机划分导致同一被试泄漏到训练和验证集。

## 可解释性
解释结果需要同时关注通道重要性、时间重要性和结论边界，不能把模型关注区域直接等同于因果脑区。
