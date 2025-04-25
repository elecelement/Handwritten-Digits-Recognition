from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """

    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    self.layers.append(Dropout(0.2))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            # print(layer)
            grads = layer.backward(grads)
            # print(grads[0][:5])
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i + 2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                                   'lambda': layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """

    def __init__(self, conv_size_list=None, linear_size_list=None, image_info=[32, 1, 28, 28], act_func=None, lambda_list=None,
                 kernel_size=3):
        self.image_info = image_info
        self.size_list = conv_size_list
        self.act_func = act_func
        self.kernel_size = kernel_size

        if conv_size_list is not None and act_func is not None:
            self.layers = []
            layer = conv2D(shape=self.image_info, in_channels=conv_size_list[0],out_channels=conv_size_list[1], kernel_size=kernel_size)
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[0]
            if act_func == 'Logistic':
                raise NotImplementedError
            elif act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            self.layers.append(layer_f)
            self.layers.append(Pool2D(size=2, stride=2, mode="max"))
            self.layers.append(Flatten())
        # 全连接层
        if linear_size_list is not None and act_func is not None:
            for i in range(len(linear_size_list) - 1):
                layer = Linear(in_dim=linear_size_list[i], out_dim=linear_size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(linear_size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # image_infp:[batch, channels, H, W]
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        new_X = X.reshape(X.shape[0], self.image_info[2], self.image_info[3])
        new_X = new_X[:, np.newaxis, :, :]
        if self.image_info[1] == 3:
            # If it's an RGB image, replicate the grayscale image 3 times
            new_X = np.repeat(new_X, 3, axis=1)
        outputs = new_X
        for layer in self.layers:
            #print(layer)
            outputs = layer(outputs)
        #print("done")
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)

        # 获取结构信息
        self.size_list = param_list[0]  # 获取size_list（包括卷积层通道数）
        self.act_func = param_list[1]  # 获取激活函数类型
        self.layers = []

        param_idx = 2  # 从第三个元素开始是每层的参数

        # 加载卷积层部分
        for i in range(len(self.size_list) - 1):
            layer_params = param_list[param_idx]

            if isinstance(layer_params, dict) and 'W' in layer_params:  # 卷积层
                layer = conv2D(
                    shape=self.image_info,
                    in_channels=self.size_list[i],
                    out_channels=self.size_list[i + 1],
                    kernel_size=self.kernel_size
                )
                layer.W = layer_params['W']
                layer.b = layer_params['b']
                layer.params = {'W': layer.W, 'b': layer.b}
                layer.weight_decay = layer_params.get('weight_decay', False)
                layer.weight_decay_lambda = layer_params.get('lambda', 0)
                self.layers.append(layer)

                # 添加激活函数
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'Logistic':
                    raise NotImplementedError("Logistic activation not implemented")

                # 添加池化层
                self.layers.append(Pool2D(size=2, stride=2, mode="max"))
                self.layers.append(Flatten())

            param_idx += 1  # 移动到下一个层

        # 加载全连接层部分
        for param in param_list[param_idx:]:
            if isinstance(param, dict) and 'in_dim' in param:  # 如果是线性层
                layer = Linear(in_dim=param['in_dim'], out_dim=param['out_dim'])
                layer.W = param['W']
                layer.b = param['b']
                layer.params = {'W': layer.W, 'b': layer.b}
                layer.weight_decay = param.get('weight_decay', False)
                layer.weight_decay_lambda = param.get('lambda', 0)
                self.layers.append(layer)

                # 添加激活函数
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'Logistic':
                    raise NotImplementedError

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]

        # 保存卷积层和全连接层的参数
        for layer in self.layers:
            if layer.optimizable:
                if isinstance(layer, Linear):  # 处理全连接层
                    param_list.append({
                        'in_dim': layer.in_dim,
                        'out_dim': layer.out_dim,
                        'W': layer.params['W'],
                        'b': layer.params['b'],
                        'weight_decay': layer.weight_decay,
                        'lambda': layer.weight_decay_lambda
                    })
                elif isinstance(layer, conv2D):  # 处理卷积层
                    param_list.append({
                        'W': layer.params['W'],
                        'b': layer.params['b'],
                        'weight_decay': layer.weight_decay,
                        'lambda': layer.weight_decay_lambda
                    })

        # 保存到文件
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)







