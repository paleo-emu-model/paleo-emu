
from src.load import load_forcing_data


def run_prediction(pipeline, pca_model, forcing_cfg, spatial_shape):
    """
    使用训练好的 pipeline 和 PCA 对预测输入进行模拟预测。

    参数：
        pipeline: 已训练的 sklearn Pipeline
        pca_model: 已拟合的 PCA 模型
        X_pred: (n_samples, 5)，预测输入
        spatial_shape: (lat, lon)，原始空间结构

    返回：
        Y_pred: 模拟预测结果，形状为 (n_samples, lat, lon)
    """
    # 内部调用加载模块
    X_pred = load_forcing_data(forcing_cfg)
    Y_pca_pred = pipeline.predict(X_pred)
    Y_full = pca_model.inverse_transform(Y_pca_pred)
    n = Y_full.shape[0]
    lat, lon = spatial_shape
    Y_out = Y_full.reshape(n, lat, lon)
    return Y_out