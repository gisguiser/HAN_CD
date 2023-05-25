# -*- coding: utf-8 -*-

import dgl
import torch
import networkx as nx

import matplotlib.pyplot as plt

#——————————————————————参考SCD构建的异构图网络————————————————————————————————————————
def build_graph(type):
    if type == 'stu':
        data_dict = {}
        edge_list = []
        with open('./data/self/graph/u_from_e.txt', 'r') as f:
            #exer_id=line[0],stu_id=line[1]
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]),int(line[1])))
        src, dst = tuple(zip(*edge_list))
        data_dict[('exer', 'es', 'stu')] = (
                 torch.tensor(src), torch.tensor(dst))
        data_dict[('stu', 'se', 'exer')] = (
                 torch.tensor(dst), torch.tensor(src))
        g = dgl.heterograph(data_dict)
        return g
    elif type == 'know':
        data_dict = {}
        edge_list = []
        with open('./data/self/graph/e_from_k.txt', 'r') as f:
            #src:kn_id=line[0]     ,       dst:exer_id=line[1]
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        data_dict[('know','ke','exer')]=(
            torch.tensor(src),torch.tensor(dst))
        data_dict[('exer', 'ek', 'know')] = (
            torch.tensor(dst), torch.tensor(src))
        g = dgl.heterograph(data_dict)
        return  g
    elif type == 'exer':
        data_dict = {}
        edge_list = []
        with open('./data/self/graph/u_from_e.txt', 'r') as f:
            # exer_id=line[0],stu_id=line[1]
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        data_dict[('exer', 'es', 'stu')] = (
            torch.tensor(src), torch.tensor(dst))
        data_dict[('stu', 'se', 'exer')] = (
            torch.tensor(dst), torch.tensor(src))

        edge_list = []
        with open('./data/self/graph/e_from_k.txt', 'r') as f:
            # src:kn_id=line[0]     ,       dst:exer_id=line[1]
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        data_dict[('know', 'ke', 'exer')] = (
            torch.tensor(src), torch.tensor(dst))
        data_dict[('exer', 'ek', 'know')] = (
            torch.tensor(dst), torch.tensor(src))
        g = dgl.heterograph(data_dict)
        return g


#——————————————————————通过字典集构建的异构图网络————————————————————————————————————
#构建异构图网络,一共三类异构图网络meta-path
"""def build_graph(type, node):
     #构建SES：学生-练习-学生的元路径图
    if type == 'stu':
        # 读取文件并创建节点列表
        exer_ids = []
        stu_ids = []
        with open('./data/ASSIST/graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                exer_id = int(line[0])
                stu_id = int(line[1])
                exer_ids.append(exer_id)
                stu_ids.append(stu_id)
        # 为每个练习节点分配一个唯一的ID
        node_dict1 = {}
        node_id1 = 0
        for id in exer_ids:
            if id not in node_dict1:
                node_dict1[id] = node_id1
                node_id1 += 1
        # 为每个练习节点分配一个唯一的ID
        node_dict2 = {}
        node_id2 = 0
        for id in stu_ids:
            if id not in node_dict2:
                node_dict2[id] = node_id2
                node_id2 += 1
        # 创建异构图
        g = dgl.heterograph(
            {
                ("stu", "se", "exer"): (
                    torch.tensor([node_dict2[i] for i in stu_ids]), torch.tensor([node_dict1[i] for i in exer_ids])),
                ("exer", "es", "stu"): (
                    torch.tensor([node_dict1[i] for i in exer_ids]), torch.tensor([node_dict2[i] for i in stu_ids]))
            }
        )
        return g

    #构建KEK：概念-练习-概念的元路径图
    elif type == 'know':
        # 读取文件并创建节点列表
        exer_ids = []
        kn_ids = []
        with open('./data/ASSIST/graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                kn_id = int(line[0])
                exer_id = int(line[1])
                exer_ids.append(exer_id)
                kn_ids.append(kn_id)

        # 为每个练习节点分配一个唯一的ID
        node_dict1 = {}
        node_id1 = 0
        for id in exer_ids:
            if id not in node_dict1:
                node_dict1[id] = node_id1
                node_id1 += 1
        # 为每个概念节点分配一个唯一的ID
        node_dict2 = {}
        node_id2 = 0
        for id in kn_ids:
            if id not in node_dict2:
                node_dict2[id] = node_id2
                node_id2 += 1
        g = dgl.heterograph(
            {
                ("know", "ke", "exer"): (
                    torch.tensor([node_dict2[i] for i in kn_ids]), torch.tensor([node_dict1[i] for i in exer_ids])),
                ("exer", "ek", "know"): (
                    torch.tensor([node_dict1[i] for i in exer_ids]), torch.tensor([node_dict2[i] for i in kn_ids])),
            }
        )
        return g

     #构建ESE（练习-学生-练习）和 EKE（练习-概念-练习）两条元路径的图
    elif type == 'exer':
        exer_dic1 = []
        stu_dic1 = []
        with open('./data/ASSIST/graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                exer_id1 = int(line[0])
                stu_id = int(line[1])
                exer_dic1.append(exer_id1)
                stu_dic1.append(stu_id)
        # 为ESE元路径中每个练习节点分配一个唯一的ID
        node_dict1 = {}
        node_id1 = 0
        for id in exer_dic1:
            if id not in node_dict1:
                node_dict1[id] = node_id1
                node_id1 += 1
        # 为为ESE元路径中每个学生节点分配一个唯一的ID
        node_dict2 = {}
        node_id2 = 0
        for id in stu_dic1:
            if id not in node_dict2:
                node_dict2[id] = node_id2
                node_id2 += 1

        exer_dict2 = []
        kn_dic = []
        with open('./data/ASSIST/graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                exer_id2 = int(line[1])
                kn_id = int(line[0])
                exer_dict2.append(exer_id2)
                kn_dic.append(kn_id)
        # 为EKE元路径中每个练习节点分配一个唯一的ID
        node_dict3 = {}
        node_id3 = 0
        for id in exer_dict2:
            if id not in node_dict3:
                node_dict3[id] = node_id3
                node_id3 += 1
        # 为EKE元路径中每个概念节点分配一个唯一的ID
        node_dict4 = {}
        node_id4 = 0
        for id in kn_dic:
            if id not in node_dict4:
                node_dict4[id] = node_id4
                node_id4 += 1

        g = dgl.heterograph(
            {
                ("stu", "se", "exer"):
                    (torch.tensor([node_dict2[i] for i in stu_dic1]), torch.tensor([node_dict1[i] for i in exer_dic1])),
                ("exer", "es", "stu"):
                    (torch.tensor([node_dict1[i] for i in exer_dic1]), torch.tensor([node_dict2[i] for i in stu_dic1])),
                ("know", "ke", "exer"):
                    (torch.tensor([node_dict4[i] for i in kn_dic]), torch.tensor([node_dict3[i] for i in exer_dict2])),
                ("exer", "ek", "stu"):
                    (torch.tensor([node_dict3[i] for i in exer_dict2]), torch.tensor([node_dict4[i] for i in kn_dic]))
            }
        )

        return g
"""