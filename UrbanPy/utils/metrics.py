import numpy as np

def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_RMSE(pred, real):
    return np.sqrt(np.mean(np.power(real - pred, 2)))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))

def get_MRE(pred, real, upscale_factor=4):
    ori_real = real.copy()
    real[real == 0] = 1
    return np.mean(np.abs(ori_real - pred) / real)
