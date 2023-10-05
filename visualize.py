""" Network architecture visualizer using graphviz """
import pygraphviz as pgv
from models import gp_tree
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def plot_fitness_curve(vis_items, plt_path):
    plt.figure()
    y1 = vis_items["min"]
    y2 = vis_items["max"]
    y3 = vis_items["avg"]
    x = [i for i in range(len(y1))]  # 点的横坐标

    plt.plot(x, y1, marker='o', color='r', label='min')
    plt.plot(x, y2, marker='o', color='b', label='max')
    plt.plot(x, y3, marker='o', color='g', label='average')

    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.legend()

    plt.savefig(os.path.join(plt_path, "fitness_curve.png"))
    # plt.show()


def plot_fitness_boxplot(vis_items, plt_path):
    plt.figure()
    data = vis_items["fit"]

    plt.boxplot(data, vert=True, patch_artist=True)

    x_positions = np.arange(len(data))  # 生成从1到数据组数的数组
    x_labels = ['{}'.format(i) for i in range(len(data))]  # 创建标签

    plt.xticks(x_positions, x_labels)
    plt.title('Fitness Box Plot')
    plt.xlabel('generation')
    plt.ylabel('fitness')

    plt.savefig(os.path.join(plt_path, "fitness_boxplot.png"))


def plot_histogram_of_dict(input_dict, save_dir):
    plt.figure()
    # 提取字典中的所有值
    values = list(input_dict.values())

    # 使用Counter来统计不同值出现的次数
    value_counts = Counter(values)

    # 将值和对应的计数分开
    # unique_values = list(value_counts.keys())
    unique_values = [str(v) for v in value_counts]
    counts = list(value_counts.values())

    # 绘制直方图
    plt.bar(unique_values, counts)
    plt.xlabel('Operation')
    plt.ylabel('Count')
    plt.title('Operation Count')

    # 保存直方图到文件
    plt.savefig(os.path.join(save_dir, "operation_histogram.png"), bbox_inches='tight')


def draw_graph(expr, save_dir, save_name):
    nodes, edges, labels = graph(expr)

    plot_histogram_of_dict(labels, save_dir)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(os.path.join(save_dir, save_name))


def graph(expr):
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, gp_tree.Primitive) else node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels
