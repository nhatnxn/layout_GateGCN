
import torch
import numpy as np
from PIL import Image
import streamlit as st
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import configs as cf
from backend.kie.kie_utils import *
from backend.kie.kie_utils import (
    load_gate_gcn_net,
    run_predict,
    vis_kie_pred,
    postprocess_scores,
    postprocess_write_info,
    prepare_graph,
    prepare_data,
    prepare_pipeline
)

import dgl

from backend.backend_utils import create_merge_cells, get_request_api

st.set_page_config(layout="wide")

import json
import os
import time

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    gcn_net = load_gate_gcn_net(cf.device, cf.kie_weight_path)
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["cnn"]["pretrained"] = False
    config["device"] = cf.device
    config["predictor"]["beamsearch"] = False
    detector = Predictor(config)

    return gcn_net, detector

gcn_net, detector = load_model()

with open('preprocessed_data/train_dataset_extra.json', 'r') as f:
    train_data = json.load(f)

with open('preprocessed_data/val_dataset_extra.json', 'r') as f:
    val_data = json.load(f)

trainset = []
valset = []
for i, key in enumerate(train_data.keys()):
    json_res = train_data[key]
    pil_img = Image.open(os.path.join('preprocessed_data/images', key))
    img = np.array(pil_img)
    group_ids = np.array([i["group_id"] for i in json_res["cells"]])
    # print('GROUP_IDS: ', group_ids)

    # print('GROUP_ID SHAPE: ', group_ids.shape[0])
    if group_ids.shape[0] < 2:
        continue
    
    cells = json_res["cells"]
    merged_cells = create_merge_cells(
        detector, img, cells, group_ids, merge_text=cf.merge_text
    )
    # print('MERGE CELLS: ', merged_cells)
    # if i <700:
    trainset.append(merged_cells)
    # else:
    #     valset.append(merged_cells)
    
    # batch_scores, boxes = run_predict(gcn_net, merged_cells, device=cf.device)
    # print('BATCH_SCOERS: ', batch_scores)
    
    # values, preds = postprocess_scores(
    #     batch_scores, score_ths=cf.score_ths, get_max=cf.get_max
    # )
    # print('VALUE: ', values)
    # print('PREDS: ', preds)
    # if i == 99:
    #     break

for i, key in enumerate(val_data.keys()):
    json_res = val_data[key]
    pil_img = Image.open(os.path.join('preprocessed_data/images', key))
    img = np.array(pil_img)
    group_ids = np.array([i["group_id"] for i in json_res["cells"]])
    # print('GROUP_IDS: ', group_ids)
    
    # print('GROUP_IDS: ', group_ids.shape)
    if group_ids.shape[0] < 2:
        continue

    cells = json_res["cells"]
    merged_cells = create_merge_cells(
        detector, img, cells, group_ids, merge_text=cf.merge_text
    )
    # print('MERGE CELLS: ', merged_cells)
    # if i <700:
    valset.append(merged_cells)

print('LEN: ', len(trainset), len(valset))

def prepare_data(cells, text_key="vietocr_text", text_cate='cat_id'):
    texts = []
    text_lengths = []
    polys = []
    labels = []

    for cell in cells:
        text = cell[text_key]
        text_encode = make_text_encode(text)
        text_lengths.append(text_encode.shape[0])
        texts.append(text_encode)

        label = cell[text_cate]
        # if label == 'OTHER':
        #     label = 0
        # if label == 'ADDRESS':
        #     label = 1
        # elif label == 'SELLER':
        #     label = 2
        # elif label == 'TIMESTAMP':
        #     label = 3
        # elif label == 'TOTAL_COST':
        #     label = 4
        # else: 
        #     label = 0
        # if label == '0':
        #     label = 0
        # if label == '1':
        #     label =1
        # if label == 3 or label == '3':
        #     label = 2
        if label == 5 or label == '5':
            label = 3
        labels.append(label)
        
        poly = copy.deepcopy(cell["poly"].tolist())
        poly.append(np.max(poly[0::2]) - np.min(poly[0::2]))
        poly.append(np.max(poly[1::2]) - np.min(poly[1::2]))
        poly = list(map(int, poly))
        polys.append(poly)

    texts = np.array(texts, dtype=object)
    text_lengths = np.array(text_lengths)
    polys = np.array(polys)
    return texts, text_lengths, polys, labels

def prepare_graph(cells):

    # print('CELLS: ', cells)
    _g, _boxes, _edge_data, _snorm_n, _snorm_e, _texts, _text_length, _origin_boxes, _graph_node_size, _graph_edge_size = [],[],[],[],[],[],[],[],[],[]
    batch_labels = []
    _tab_sizes_e = []
    _tab_snorm_e = []
    _tab_sizes_n = []
    _tab_snorm_n = []

    for cell in cells:
        alr = 0
        # print('CELL: ', cell)
        texts, text_lengths, boxes, labels = prepare_data(cell)
        # print('TEXTS: ', texts)
        # print('TEXT_LENGTHS: ', text_lengths)
        # print('BOXES: ', boxes)
        # print('LABELS: ', labels)
        labels = torch.tensor(labels)
        

        # print('TEXTS: ', texts)
        # print('TEXT_LENGTHS: ', text_lengths)
        # print('BOXES: ', boxes)
        # print('LEN BOXES: ', len(boxes))
        # print('LABELS: ', labels)
        # print('LEN LABELS: ', len(labels))

        origin_boxes = boxes.copy()
        node_nums = text_lengths.shape[0]

        src = []
        dst = []
        edge_data = []
        
        # print('NODE_NUMS: ', node_nums)
        # print('BOXES: ', boxes)
        for i in range(node_nums):
            for j in range(node_nums):
                if i == j:
                    continue

                edata = []
                # y distance
                y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                w = boxes[i, 8]
                h = boxes[i, 9]
                # print('H: ', h)
                # print('Y_DISTANCE: ', y_distance)
                if np.abs(y_distance) > 3 * h:
                    continue

                if h < 13 or w < 13:
                    alr+=1
                    continue

                x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                edata.append(y_distance)
                edata.append(x_distance)

                edge_data.append(edata)
                src.append(i)
                dst.append(j)

        # print('SRC: ', src)
        # print('DST: ', dst)
        # print('EDGE_DATA: ', edge_data)

        edge_data = np.array(edge_data)
        g = dgl.DGLGraph()
        g.add_nodes(node_nums)
        g.add_edges(src, dst)

        if not edge_data.shape[0]:
            continue

        boxes, edge_data, text, text_length = prepare_pipeline(
            boxes, edge_data, texts, text_lengths
        )
        boxes = torch.from_numpy(boxes).float()
        edge_data = torch.from_numpy(edge_data).float()

        tab_sizes_n = g.number_of_nodes()
        tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1.0 / float(tab_sizes_n))
        _tab_sizes_n.append(tab_sizes_n)
        _tab_snorm_n.append(tab_snorm_n)
        # snorm_n = tab_snorm_n.sqrt()

        tab_sizes_e = g.number_of_edges()
        tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1.0 / float(tab_sizes_e))
        _tab_sizes_e.append(tab_sizes_e)
        _tab_snorm_e.append(tab_snorm_e)
        # snorm_e = tab_snorm_e.sqrt()

        # print('TEXT_LENGTHS: ', text_lengths)
        # max_length = text_lengths.max()
        max_length = 256
        # print('TEXT: ', text)
        new_text = [
            np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), "constant"), axis=0)
            for t in text
        ]
        # print('TEXTS: ', new_text[:3])
        texts = np.concatenate(new_text)

        texts = torch.from_numpy(np.vstack(texts).astype(np.int))
        # print('TEXT_LENGTH: ', text_length)
        text_length = torch.from_numpy(np.array(text_length))

        # graph_node_size = [g.number_of_nodes()]
        # graph_edge_size = [g.number_of_edges()]

        _g.append(g)
        _boxes.append(boxes)
        _edge_data.append(edge_data)
        # _snorm_n.append(snorm_n)
        # _snorm_e.append(snorm_e)
        _texts.append(texts)
        _text_length.append(text_length)
        _origin_boxes.append(origin_boxes)
        # _graph_node_size.append(graph_node_size)
        # _graph_edge_size.append(graph_edge_size)

        batch_labels.append(labels)
    # print('_BOXES: ', _boxes)
    
    # _graph_node_size = torch.cat(_graph_node_size)
    # _graph_edge_size = torch.cat(_graph_edge_size)
    _boxes = torch.cat(_boxes)
    _edge_data = torch.cat(_edge_data)
    _snorm_n = torch.cat(_tab_snorm_n).sqrt()
    _snorm_e = torch.cat(_tab_snorm_e).sqrt()
    _g = dgl.batch(_g)

    _graph_node_size = [_g.number_of_nodes()]
    _graph_edge_size = [_g.number_of_edges()]

    # print('_GRAPH_NODE_SIZE: ', _graph_node_size)

    # print('_TEXT: ', _texts)
    # print('_TEXT SHAPE: ', _texts[0].shape, _texts[1].shape)
    _texts = torch.cat(_texts)
    _text_length = torch.cat(_text_length)
    batch_labels = torch.cat(batch_labels)
    # print('BATCH_LABELS: ', batch_labels)
    # print('ALR: ', alr)

    return (
        _g,
        _boxes,
        _edge_data,
        _snorm_n,
        _snorm_e,
        _texts,
        _text_length,
        _origin_boxes,
        _graph_node_size,
        _graph_edge_size,
    ), batch_labels

train_loader = DataLoader(trainset, batch_size=10, shuffle=True, collate_fn=prepare_graph)
val_loader = DataLoader(valset, batch_size=10, shuffle=False, collate_fn=prepare_graph)
# print(list(train_loader)[0])

optimizer = torch.optim.Adam(gcn_net.parameters(), lr=0.0001)

def run_predict(gcn_net, merged_cells, device="cpu"):

    batch_graphs, batch_labels = prepare_graph(merged_cells)

    # print('DETAIL: ',
    #     batch_graphs,
    #     batch_x,
    #     batch_e,
    #     batch_snorm_n,
    #     batch_snorm_e,
    #     text,
    #     text_length,
    #     boxes,
    #     graph_node_size,
    #     graph_edge_size,
    # )

    batch_labels = batch_labels.to(device)
    batch_graphs = batch_graphs.to(device)
    batch_x = batch_x.to(device)
    batch_e = batch_e.to(device)

    text = text.to(device)
    text_length = text_length.to(device)
    batch_snorm_e = batch_snorm_e.to(device)
    batch_snorm_n = batch_snorm_n.to(device)

    batch_graphs = batch_graphs.to(device)
    # print('GRAPH NOTE SIZE: ', graph_node_size)
    # print('GRAPH EDGE SIZE: ', graph_edge_size)
    batch_scores = gcn_net.forward(
        batch_graphs,
        batch_x,
        batch_e,
        text,
        text_length,
        batch_snorm_n,
        batch_snorm_e,
        graph_node_size,
        graph_edge_size,
    )

    # print('BATCH_SCORES: ', batch_scores)
    # print('BOXES: ', boxes)
    return batch_scores, boxes

def train_one_epoch(net, data_loader):
    """
    train one epoch
    """
    net.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        # batch_x = batch_graphs.ndata['feat']
        # batch_e = batch_graphs.edata['feat']
        # batch_snorm_n = batch_snorm_n
        # batch_snorm_e = batch_snorm_e
        # batch_labels = batch_labels
        (
        batch_g,
        batch_x,
        batch_e,
        batch_snorm_n,
        batch_snorm_e,
        text,
        text_length,
        boxes,
        graph_node_size,
        graph_edge_size,
        ) = batch_graphs

        # print('BATCH_GRAPHS: ', batch_g)
        # print('BATCH_LABELS: ', batch_labels)

        batch_labels = batch_labels.to(cf.device)

        batch_scores = net.forward(batch_g,
        batch_x,
        batch_e,
        text,
        text_length,
        batch_snorm_n,
        batch_snorm_e,
        graph_node_size,
        graph_edge_size,)
        # print('BATCH_SCORES: ', batch_scores)
        # print('BATCH_SCORES SHAPE: ', batch_scores.shape)
        # print('BATCH_LABELS: ', batch_labels)
        # print('BATCH_LABELS SHAPE: ', batch_labels.shape)
        loss = net.loss(batch_scores, batch_labels)
        # print('LOSS: ', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += net.accuracy(batch_scores,batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    return epoch_loss, epoch_train_acc  



def evaluate_network(net, data_loader):
    """
    evaluate test set
    """
    net.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            # batch_x = batch_graphs.ndata['feat']
            # batch_e = batch_graphs.edata['feat']
            # batch_snorm_n = batch_snorm_n
            # batch_snorm_e = batch_snorm_e
            # batch_labels = batch_labels
            (
            batch_g,
            batch_x,
            batch_e,
            batch_snorm_n,
            batch_snorm_e,
            text,
            text_length,
            boxes,
            graph_node_size,
            graph_edge_size,
            ) = batch_graphs

            # print('BATCH_GRAPHS: ', batch_g)
            # print('BATCH_LABELS: ', batch_labels)

            batch_labels = batch_labels.to(cf.device)

            batch_scores = net.forward(batch_g,
            batch_x,
            batch_e,
            text,
            text_length,
            batch_snorm_n,
            batch_snorm_e,
            graph_node_size,
            graph_edge_size,)
            loss = net.loss(batch_scores, batch_labels)
            # print('LOSS: ', loss)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += net.accuracy(batch_scores,batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
    return epoch_test_loss, epoch_test_acc
    


epoch_train_losses = []
epoch_test_losses = []
epoch_val_losses = []
epoch_train_accs = []
epoch_test_accs = []
epoch_val_accs = []
val_acc = 0
for epoch in range(50):
    
    start = time.time()
    epoch_train_loss, epoch_train_acc = train_one_epoch(gcn_net, train_loader)
    # epoch_test_loss, epoch_test_acc = evaluate_network(net, test_loader)
    epoch_val_loss, epoch_val_acc = evaluate_network(gcn_net, val_loader)
    
    if epoch_val_acc > val_acc:
        print('saving ...')
        torch.save(gcn_net.state_dict(), 'best.pkl')

    print('Epoch {}, time {:.4f}, train_loss: {:.4f}, val_loss: {:.4f} \n                     train_acc: {:.4f}, val_acc: {:.4f}'.format(epoch, time.time()-start, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc))
    