import torch
import torch.nn as nn
from models import ops
import gp_tree


class ParallelSequentialModule(nn.ModuleList):
    def __init__(self, modules=None, parallel=True):
        super(ParallelSequentialModule, self).__init__(modules)
        self.parallel = parallel

    def is_parallel(self):
        return self.parallel


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, expr, n_classes=2):
        super().__init__()
        nodes, edges, labels, sign, input_nodes = self.create_graph(expr)
        C_cur = 16     # 当前Sequential模块的输出通道数
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),    # 前三个参数分别是输入图片的通道数，卷积核的数量，卷积核的大小
            nn.BatchNorm2d(C_cur)
            # BatchNorm2d 对 minibatch 3d 数据组成的 4d 输入进行batchnormalization操作，num_features为(N,C,H,W)的C
        )

        templates = {inn: [] for inn in input_nodes}

        # create input
        self.dag = nn.ModuleList()
        while len(templates) > 0:
            new_templates = {}
            for i_node in templates.keys():
                count = 0
                sub_dag = ParallelSequentialModule(parallel=False)
                c_node = i_node
                c_label = labels[c_node]
                while "Root" not in c_label and ("Add" not in c_label or count == 0):  # not output and add
                    if count == 0:
                        # ignore the first times
                        count = 1
                    for edge in edges:
                        if c_node == edge[1]:
                            c_node = edge[0]
                            c_label = labels[c_node]
                            if "Root" not in c_label and "Add" not in c_label:
                                op = ops.MixedOp(c_label, C_cur)
                                sub_dag.append(op)
                            elif "Add" in c_label:
                                # Add
                                if c_node not in new_templates.keys():
                                    new_templates[c_node] = ParallelSequentialModule(parallel=True)
                                new_templates[c_node].append(sub_dag)
                            else:
                                # Root
                                self.dag.append(sub_dag)
                            break
            templates = new_templates

        C_cur = C_cur * len(self.dag)

        self.gap = nn.AdaptiveAvgPool2d(1)  # 构建一个平均池化层，output size是1x1
        self.linear = nn.Linear(C_cur, n_classes)

    def create_graph(self, expr):
        nodes = list(range(len(expr)))
        edges = list()
        labels = dict()
        sign = list()
        input_nodes = list()

        stack = []
        for i, node in enumerate(expr):
            if stack:
                edges.append((stack[-1][0], i))
                stack[-1][1] -= 1
            labels[i] = node.name if isinstance(node, gp_tree.Primitive) else node.value
            if isinstance(node, gp_tree.Primitive) and "Root" in node.name:
                sign.append(0)
            elif isinstance(node, gp_tree.Primitive) and "Add" in node.name:
                sign.append(1)
            elif isinstance(node, gp_tree.Terminal):
                # image0 : input
                sign.append(2)
                input_nodes.append(i)
            else:
                sign.append(3)
            stack.append([i, node.arity])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return nodes, edges, labels, sign, input_nodes

    def recursive_forward(self, x, sub_net):
        inner_sum = None
        for module in sub_net:
            if isinstance(module, ParallelSequentialModule) and module.is_parallel():
                inner_result = self.recursive_forward(x, module)
            else:
                inner_result = x
                for cell in module:
                    if isinstance(cell, ParallelSequentialModule) and module.is_parallel():
                        inner_result = self.recursive_forward(inner_result, cell)
                    else:
                        inner_result = cell(inner_result)

            if inner_sum is None:
                inner_sum = inner_result
            else:
                inner_sum += inner_result
        return inner_sum if inner_sum is not None else x

    # 前向传播时自动调用, cell中的计算过程
    def forward(self, x):
        s_cur = self.stem(x)
        states = []
        for sub_net in self.dag:
            if isinstance(sub_net, ParallelSequentialModule) and sub_net.is_parallel():
                s0 = self.recursive_forward(s_cur, sub_net)
            else:
                s0 = s_cur
                for cell in sub_net:
                    if isinstance(cell, ParallelSequentialModule) and sub_net.is_parallel():
                        s0 = self.recursive_forward(s0, cell)
                    else:
                        s0 = cell(s0)
            states.append(s0)
        s1 = torch.cat(states, dim=1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


if __name__ == '__main__':
    test = "Root3(MaxP(Conv5(Add(Image0, Image0))), AveP(MaxP(Add(Image0, Image0))), Conv5(AveP(Add(Image0, Image0))))"
    sc = SearchCell(test)
