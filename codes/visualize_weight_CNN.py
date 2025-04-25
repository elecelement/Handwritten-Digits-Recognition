import matplotlib.pyplot as plt
import numpy as np
import mynn as nn


def visualize_kernels(kernels, grid_size=None, figsize=(10, 10)):
    """
    可视化卷积核（filters）。

    kernels: numpy 数组，形状为 (out_channels, in_channels, kernel_size, kernel_size)
    grid_size: 网格布局，指定行和列的数量，默认根据卷积核数量自动计算
    figsize: 图形大小，调整显示大小
    """
    num_filters = kernels.shape[0]  # 获取卷积核的数量
    if grid_size is None:
        # 根据卷积核数量动态计算网格的行和列数
        grid_size = (int(np.sqrt(num_filters))+1, int(np.ceil(num_filters / np.sqrt(num_filters))))
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.ravel()

    # 如果卷积核数量小于网格位置数，关闭多余的子图
    for i in range(num_filters):
        kernel = kernels[i, 0]  # 选择第一个输入通道的卷积核（因为你有 1 个输入通道）

        axes[i].imshow(kernel, cmap='gray')
        axes[i].axis('off')  # 关闭坐标轴
        axes[i].set_title(f'Filter {i + 1}')

    # 关闭未使用的子图
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_saved_model_kernels(save_path):
    """
    从已保存的 CNN 模型中加载卷积核并进行可视化。
    """
    # 创建一个空模型实例
    model = nn.models.Model_CNN()  # 假设你已经定义了模型类

    # 加载保存的模型
    model.load_model(save_path)

    # 提取卷积层的卷积核并进行可视化
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):  # 确保是卷积层
            filters = layer.params['W']  # 提取卷积核（权重）

            # 打印卷积核的形状，帮助调试
            print(f"Filter shape: {filters.shape}")

            visualize_kernels(filters)  # 可视化卷积核

save_path = r".\saved_models\best_model_CNN.pickle"
visualize_saved_model_kernels(save_path)