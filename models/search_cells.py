import torch
import torch.nn as nn
from models.ops import OperationSelector
from models import gp_tree


class ParallelSequentialModule(nn.ModuleList):
    def __init__(self, modules=None, parallel=True):
        super(ParallelSequentialModule, self).__init__(modules)
        self.parallel = parallel
        self.w_list = list()

    def is_parallel(self):
        return self.parallel


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, expr, operations_type, n_classes=2, num_hidden_layers=16, input_channels=1):
        super().__init__()
        nodes, edges, labels, input_nodes, w_nodes = self.create_graph(expr)
        C_cur = num_hidden_layers    # 当前Sequential模块的输出通道数
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_cur, 3, 1, 1, bias=False),    # 前三个参数分别是输入图片的通道数，卷积核的数量，卷积核的大小
            nn.BatchNorm2d(C_cur)
            # BatchNorm2d 对 minibatch 3d 数据组成的 4d 输入进行batchnormalization操作，num_features为(N,C,H,W)的C
        )

        templates = {inn: None for inn in input_nodes}

        # create input
        self.dag = nn.ModuleList()
        while len(templates) > 0:
            new_templates = {}
            next_templates = {}
            add_input_node = {}
            add_num = {}
            for i_node, i_value in templates.items():
                count = 0
                sub_dag = ParallelSequentialModule(parallel=False)
                if i_value is not None:
                    sub_dag.append(i_value)
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
                                op = OperationSelector(c_label, C_cur, operations_type)
                                sub_dag.append(op)
                            elif "Add" in c_label:
                                # Add
                                if c_node not in new_templates.keys():
                                    new_templates[c_node] = ParallelSequentialModule(parallel=True)
                                    add_input_node[c_node] = i_node
                                    add_num[c_node] = int(c_label.replace("Add", "")) if c_label != "Add" else 2
                                new_templates[c_node].append(sub_dag)
                            else:
                                # Root
                                self.dag.append(sub_dag)
                            break
            for n_k, n_w in new_templates.items():
                if len(n_w) == add_num[n_k]:
                    next_templates[n_k] = n_w
                else:
                    next_templates[add_input_node[n_k]] = None

            if len(next_templates) > 0:
                for add_node, add_module in next_templates.items():
                    if add_module is not None:
                        for edge in edges:
                            if edge[0] == add_node and edge[1] in w_nodes:
                                add_module.w_list.append(labels[edge[1]])
            templates = next_templates

        C_cur = C_cur * len(self.dag)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 构建一个平均池化层，output size是1x1
        self.dropout = nn.Dropout(0.7)
        self.linear = nn.Linear(C_cur, n_classes)

    def create_graph(self, expr):
        nodes = list(range(len(expr)))
        edges = list()
        labels = dict()

        input_nodes = list()
        w_nodes = list()

        stack = []
        for i, node in enumerate(expr):
            if stack:
                edges.append((stack[-1][0], i))
                stack[-1][1] -= 1
            labels[i] = node.name if isinstance(node, gp_tree.Primitive) else node.value

            if isinstance(node, gp_tree.Terminal) and "Image" in node.name:
                input_nodes.append(i)
            elif isinstance(node, gp_tree.Terminal):
                w_nodes.append(i)

            stack.append([i, node.arity])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return nodes, edges, labels, input_nodes, w_nodes

    def recursive_forward(self, x, sub_net):
        if isinstance(sub_net, ParallelSequentialModule) and sub_net.is_parallel():
            inner_sum = None
            for idx, module in enumerate(sub_net):
                if isinstance(module, ParallelSequentialModule):
                    inner_result = self.recursive_forward(x, module)
                else:
                    inner_result = module(x)

                if inner_sum is None:
                    inner_sum = inner_result * sub_net.w_list[idx]
                else:
                    inner_sum += inner_result * sub_net.w_list[idx]
            return inner_sum

        elif isinstance(sub_net, ParallelSequentialModule):
            inner_result = x
            for idx, module in enumerate(sub_net):
                if isinstance(module, ParallelSequentialModule):
                    inner_result = self.recursive_forward(inner_result, module)
                else:
                    inner_result = module(inner_result)
            return inner_result
        return x

    # 前向传播时自动调用, cell中的计算过程
    def forward(self, x):
        s_cur = self.stem(x)
        states = []
        for sub_net in self.dag:
            if isinstance(sub_net, ParallelSequentialModule):
                s0 = self.recursive_forward(s_cur, sub_net)
            else:
                s0 = sub_net(s0)
            states.append(s0)
        s1 = torch.cat(states, dim=1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        out = self.dropout(out)
        logits = self.linear(out)
        return logits

