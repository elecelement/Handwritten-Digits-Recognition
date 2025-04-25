import pickle
import matplotlib.pyplot as plt
import numpy as np
import mynn as nn


def visualize_weights(weights, input_size=28, num_filters_per_page=16, figsize=(10, 10)):
    """
    可视化全连接层的权重（分多次展示，每次展示 num_filters_per_page 个神经元的权重）。

    weights: numpy 数组，形状为 (input_size * input_size, output_size)
    input_size: 输入图像的尺寸（假设是正方形图像）
    num_filters_per_page: 每次展示的神经元数量
    figsize: 图形大小，调整显示大小
    """
    num_weights = weights.shape[1]  # 输出神经元的数量
    num_pages = int(np.ceil(num_weights / num_filters_per_page))  # 计算需要展示的页数

    for page in range(num_pages):
        # 计算当前页面需要展示的神经元范围
        start_idx = page * num_filters_per_page
        end_idx = min((page + 1) * num_filters_per_page, num_weights)

        num_weights_in_page = end_idx - start_idx
        grid_size = (
        int(np.sqrt(num_weights_in_page)), int(np.ceil(num_weights_in_page / np.sqrt(num_weights_in_page))))

        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
        axes = axes.ravel()

        for i in range(num_weights_in_page):
            weight = weights[:, start_idx + i].reshape(input_size, input_size)  # 将权重重塑为图像形状
            axes[i].imshow(weight, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Neuron {start_idx + i + 1}')

        for i in range(num_weights_in_page, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


# 示例使用：
def visualize_saved_model_weights(save_path):
    # 创建模型实例并加载保存的模型
    model = nn.models.Model_MLP()  # 假设你使用的是 MLP 网络结构
    model.load_model(save_path)  # 加载保存的模型

    # 提取模型权重（在全连接层）
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):  # 确保是线性层
            weights = layer.params['W']  # 提取权重矩阵（784 x 256）
            visualize_weights(weights, input_size=28, num_filters_per_page=16)  # 每次展示 16 个神经元


# 示例使用：
save_path = r".\saved_models\best_model_mlp256.pickle"
visualize_saved_model_weights(save_path)
