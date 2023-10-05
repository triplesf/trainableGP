""" Operations """
import torch
import torch.nn as nn


standard_operations = {
    'AvePF': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'AveP': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'MaxPF': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'MaxP': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'Conv3F': lambda C, stride, affine: StdConv(C, C, 3, stride, 1, affine=affine),
    'Conv3': lambda C, stride, affine: StdConv(C, C, 3, stride, 1, affine=affine),
    'Conv5F': lambda C, stride, affine: StdConv(C, C, 5, stride, 2, affine=affine),
    'Conv5': lambda C, stride, affine: StdConv(C, C, 5, stride, 2, affine=affine),
}


darts_operations = {
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)
}


single_operations = {
    'MaxP': lambda C, stride, affine: nn.MaxPool2d(3, stride, 1),
    'AveP': lambda C, stride, affine: nn.AvgPool2d(3, stride, 1, count_include_pad=False),
    'RELU': lambda C, stride, affine: nn.ReLU(),
    'Conv3': lambda C, stride, affine: nn.Conv2d(C, C, 3, stride, 1, bias=False),
    'Conv5': lambda C, stride, affine: nn.Conv2d(C, C, 5, stride, 2, bias=False),
    'Conv7': lambda C, stride, affine: nn.Conv2d(C, C, 7, stride, 3, bias=False),
    'BN': lambda C, stride, affine: nn.BatchNorm2d(C, affine=affine)
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)  # count_include_pad=False表示不把填充的0计算进去
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    长方形的卷积,增加了一点特征图的长和宽
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, (padding, 0), bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, (0, padding), bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    深度可分离卷积, groups=C_in, 表示把输入特种图分成C_in(输入通道数)那么多组, 然后加C_out(输出通道数)1*1的卷积, 这样可以对每个通道单独提取特征, 同时降低了参数量和计算量。
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    深度可分离卷积, 由两个上面的深度分组卷积组成
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


# skip conncet
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """
    把特种图的输出变为全是0，但特征图的大小会根据stride而改变
    """
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    将特征图大小变为原来的一半
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class OperationSelector(nn.Module):
    """ Mixed operation """
    def __init__(self, c_labels, C, operations_type):
        super().__init__()
        if operations_type == "standard":
            self._ops = standard_operations[c_labels](C, 1, affine=False)
        elif operations_type == "darts":
            self._ops = darts_operations[c_labels](C, 1, affine=False)
        elif operations_type == "single":
            self._ops = single_operations[c_labels](C, 1, affine=False)

    def forward(self, x):
        """
        Args:
            x: input
        """
        return self._ops(x)
