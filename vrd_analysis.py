import os
import json
import numpy as np
from collections import defaultdict
import time, datetime
import matplotlib.pyplot as plt
import statistics
import plotly.graph_objects as go
import plotly.express as px
from utils import *


def count_relation_chars(groundtruth, predicate_counts, object_counts):
    length = {'S': 0, 'M': 0, 'L': 0}
    preins = {'XS': 0, 'S': 0, 'M': 0, 'L': 0, 'XL': 0}  # predicate instance
    subins = {'XS': 0, 'S': 0, 'M': 0, 'L': 0, 'XL': 0}  # subject instance
    objins = {'XS': 0, 'S': 0, 'M': 0, 'L': 0, 'XL': 0}  # object instance
    subpxl = {'XS': 0, 'S': 0, 'M': 0, 'L': 0, 'XL': 0}  # subject pixel area
    objpxl = {'XS': 0, 'S': 0, 'M': 0, 'L': 0, 'XL': 0}  # subject pixel area

    for _, all_gt in groundtruth.items():
            for gt in all_gt:
                duration = float(gt['duration'][1] - gt['duration'][0]) / 30
                if duration <= 10:
                    length['S'] += 1
                elif duration <= 20:
                    length['M'] += 1
                else:
                    length['L'] += 1

                if predicate_counts[gt['triplet'][1]] <= 10:
                    preins['XS'] += 1
                elif predicate_counts[gt['triplet'][1]] <= 100:
                    preins['S'] += 1
                elif predicate_counts[gt['triplet'][1]] <= 1000:
                    preins['M'] += 1
                elif predicate_counts[gt['triplet'][1]] <= 10000:
                    preins['L'] += 1
                else:
                    preins['XL'] += 1

                if object_counts[gt['triplet'][0]] <= 10:
                    subins['XS'] += 1
                elif object_counts[gt['triplet'][0]] <= 100:
                    subins['S'] += 1
                elif object_counts[gt['triplet'][0]] <= 1000:
                    subins['M'] += 1
                elif object_counts[gt['triplet'][0]] <= 10000:
                    subins['L'] += 1
                else:
                    subins['XL'] += 1

                if object_counts[gt['triplet'][2]] <= 10:
                    objins['XS'] += 1
                elif object_counts[gt['triplet'][2]] <= 100:
                    objins['S'] += 1
                elif object_counts[gt['triplet'][2]] <= 1000:
                    objins['M'] += 1
                elif object_counts[gt['triplet'][2]] <= 10000:
                    objins['L'] += 1
                else:
                    objins['XL'] += 1

                sub_box = gt['sub_traj'][0]
                # json_path = os.path.join('VidVRD/vidvrd-dataset/test', vid + '.json')
                # with open(json_path, 'r') as f:
                #     this_anno = json.load(f)
                # vid_h = this_anno['height']
                # vid_w = this_anno['width']
                vid_h = 576
                vid_w = 1280
                sub_area = ((sub_box[2] - sub_box[0]) * (sub_box[3] - sub_box[1])) / (vid_h * vid_w)
                if sub_area <= 0.2:
                    subpxl['XS'] += 1
                elif sub_area <= 0.4:
                    subpxl['S'] += 1
                elif sub_area <= 0.6:
                    subpxl['M'] += 1
                elif sub_area <= 0.8:
                    subpxl['L'] += 1
                else:
                    subpxl['XL'] += 1

                obj_box = gt['obj_traj'][0]
                obj_area = ((obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])) / (vid_h * vid_w)
                if obj_area <= 0.2:
                    objpxl['XS'] += 1
                elif obj_area <= 0.4:
                    objpxl['S'] += 1
                elif obj_area <= 0.6:
                    objpxl['M'] += 1
                elif obj_area <= 0.8:
                    objpxl['L'] += 1
                else:
                    objpxl['XL'] += 1
    return length, preins, subins, objins, subpxl, objpxl


viou_threshold = 0.5
min_viou_thr = 0.1
fp_error_types_legend = {
    'True Positive': 0,
    'Classification Error': 1,
    'Localization Error': 2,
    'Confusion Error': 3,
    'Double Detection Error': 4,
    'Background Error': 5
    }
fp_error_types_inverse_legend = dict([v, k] for k, v in fp_error_types_legend.items())

gt_file = 'demo_gt.json'
with open(gt_file, 'r') as f:
    groundtruth = json.load(f)
prediction_root = 'demo_det/'
file_name = 'demo'
video_ap = dict()
tot_scores = defaultdict(list)
tot_tp = defaultdict(list)
prec_at_n = defaultdict(list)
tot_gt_relations = 0
print('Analyze over {} videos...'.format(len(groundtruth)))
vid_num = len(groundtruth)

predicate_counts = {}
object_counts = {}
for k, v in groundtruth.items():
    for gt in v:
        sbj, pre, obj = gt['triplet']
        if sbj in object_counts:
            object_counts[sbj] += 1
        else:
            object_counts[sbj] = 1
        if pre in predicate_counts:
            predicate_counts[pre] += 1
        else:
            predicate_counts[pre] = 1
        if obj in object_counts:
            object_counts[obj] += 1
        else:
            object_counts[obj] = 1

fp_error_types = {}
ap_gain, average_mAP_gain = {}, {}
for err_name, err_code in fp_error_types_legend.items():
    if err_code:
        ap_gain[err_name] = {}
ap_gain['Missed GT'] = {}

vid_missed_gt = {}
# vid_all_gt = {}
prediction_id = 0
for v, vid in enumerate(sorted(groundtruth.keys())):
    gt_relations = groundtruth[vid]
    vid_missed_gt[vid] = []
    # vid_all_gt[vid].extend(gt_relations)
    print('[%d/%d] %s' % (vid_num, v + 1, vid))
    if len(gt_relations) == 0:
        continue
    tot_gt_relations += len(gt_relations)
    predict_res_path = os.path.join(prediction_root, vid + '.json')
    with open(predict_res_path) as f:
        predict_relations = json.load(f)
        predict_relations = predict_relations[vid]
    num_pred = len(predict_relations)

    predict_relations = sorted(predict_relations, key=lambda x: x['score'], reverse=True)
    predict_relations = predict_relations[:200]

    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    this_error_types = []
    hit_scores = np.ones((len(predict_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(predict_relations):
        ov_max = -float('Inf')
        k_max = -1
        top_viou = 0
        this_pred_label = pred_relation['triplet']
        for gt_idx, gt_relation in enumerate(gt_relations):
            s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                            gt_relation['sub_traj'], gt_relation['duration'])
            o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                            gt_relation['obj_traj'], gt_relation['duration'])
            ov = min(s_iou, o_iou)
            if ov > top_viou:
                top_viou = ov
                gt_with_max_tiou_label = gt_relation['triplet']
            if not gt_detected[gt_idx] and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:  # True Positive
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
            fp_error_types[prediction_id] = fp_error_types_legend['True Positive']
            this_error_types.append(fp_error_types_legend['True Positive'])
        else:  # False Positive
            if top_viou >= viou_threshold:
                if gt_with_max_tiou_label == this_pred_label:
                    # Double Detection Error
                    fp_error_types[prediction_id] = fp_error_types_legend['Double Detection Err']
                    this_error_types.append(fp_error_types_legend['Double Detection Err'])
                else:
                    # Classification Error
                    fp_error_types[prediction_id] = fp_error_types_legend['Classification Error']
                    this_error_types.append(fp_error_types_legend['Classification Error'])
            elif top_viou >= min_viou_thr:
                if gt_with_max_tiou_label == this_pred_label:
                    # Localization Error
                    fp_error_types[prediction_id] = fp_error_types_legend['Localization Error']
                    this_error_types.append(fp_error_types_legend['Localization Error'])
                else:
                    # Confusion Error
                    fp_error_types[prediction_id] = fp_error_types_legend['Confusion Error']
                    this_error_types.append(fp_error_types_legend['Confusion Error'])
            else:
                # Background Error
                fp_error_types[prediction_id] = fp_error_types_legend['Background Error']
                this_error_types.append(fp_error_types_legend['Background Error'])
        prediction_id += 1

    for gt_idx, gt_relation in enumerate(gt_relations):
        if gt_detected[gt_idx] == 0:
            vid_missed_gt[vid].append(gt_relation)

    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)

    video_ap[vid] = voc_ap(rec, prec)

    # Computes the average-mAP gain after removing each error type
    npos = len(gt_relations)

    recall = cum_tp / np.maximum(cum_tp[-1], np.finfo(np.float32).eps)
    ap_gain['Missed GT'][vid] = voc_ap(recall, prec)

    this_error_types = np.asarray(this_error_types)

    for err_name, err_code in fp_error_types_legend.items():
        if not err_code:  # TP
            continue
        tp = (this_error_types == 0)
        tp = tp[this_error_types != err_code]  # remove this error type

        fp = ~tp
        cum_tp = np.cumsum(tp).astype(np.float32)
        cum_fp = np.cumsum(fp).astype(np.float32)
        rec = cum_tp / np.maximum(npos, np.finfo(np.float32).eps)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)

        this_ap = voc_ap(rec, prec)
        ap_gain[err_name][vid] = this_ap

mean_ap = round(float(np.mean(list(video_ap.values()))), 4)
for err_name, _ in ap_gain.items():
    average_mAP_gain[err_name] = round(float(np.mean(list(ap_gain[err_name].values()))), 4) - mean_ap
# print(f'average_mAP_gain: \n{json.dumps(average_mAP_gain, sort_keys=True, indent=4)}')

fp_error_types_count = {}
fp_error_types_percent = {}
for err_type, err_id in fp_error_types_legend.items():
    fp_error_types_count[err_type] = sum(map((err_id).__eq__, fp_error_types.values()))
    fp_error_types_percent[err_type] = float(fp_error_types_count[err_type]) / len(fp_error_types)

#---------- False Positive Analysis ----------#
labels = np.asarray(['Bkg', 'Con', 'Cls', 'Loc', 'DD', 'TP'])
values = np.asarray([
    fp_error_types_percent['Background Error'],
    fp_error_types_percent['Confusion Error'],
    fp_error_types_percent['Classification Error'],
    fp_error_types_percent['Localization Error'],
    fp_error_types_percent['Double Detection Error'],
    fp_error_types_percent['True Positive']
])
order = np.asarray([0, 1, 2, 3, 4, 5])  # display order
fig = go.Figure(go.Pie(
                        values = values[order],
                        labels = labels[order],
                        texttemplate = "%{label}: %{percent:.2%f}",
                        # texttemplate = "%{label}",
                        textfont = {'size': 18},
                        direction='clockwise',
                        sort=False,
                        # color_discrete_sequence=px.colors.sequential.Plotly3)
                        marker=dict(
                            # colors=px.colors.sequential.Plotly3, 
                            colors=['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)', 'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)'], 
                                            line=dict(color='#000000', width=0)
                                    ),
                        automargin=False
                    ),
                )
fig.update_traces(showlegend=False)
fig.update_layout(margin=dict(l=10, r=10, t=60, b=0), width=350, height=430)
# fig.show()
fig.write_image(file_name+'_FP.pdf')

#---------- False Negative Analysis ----------#
labels = ['TP', 'Miss']
values = [fp_error_types_percent['True Positive'], tot_gt_relations - fp_error_types_percent['True Positive']]
donut_colors=[ 'rgba(255, 127, 14, 0.8)', 'rgba(31, 119, 180, 0.8)']
annotations=[]
annotations.append(dict(xref='paper', yref='paper',
  x=0.5, y=0.5,
  xanchor= 'center',
  yanchor='middle',
  text= '<b>Ground Truth</b>',
  font=dict(family="Arial", size=20),
  showarrow=False,
  ))

fig2 = go.Figure(data=[go.Pie(
    labels = labels,
    values = values,
    hole = .4,
    marker_colors=donut_colors,
    # texttemplate = "%{label}: %{percent:.2%f}",
    direction='clockwise',
    sort=False)])
fig2.update_traces(textposition='inside', textinfo='label+percent', showlegend=False, textfont_size=18)
fig2.update_layout(annotations=annotations)
fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=350, height=350)
# fig2.show()
fig2.write_image(file_name+'_FN.pdf')

#---------- relation characteristics analysis ----------#
gt_length, gt_preins, gt_subins, gt_objins, gt_subpxl, gt_objpxl = count_relation_chars(groundtruth, predicate_counts, object_counts)
miss_length, miss_preins, miss_subins, miss_objins, miss_subpxl, miss_objpxl = count_relation_chars(vid_missed_gt, predicate_counts, object_counts)
figsize = (20, 3.5)
fontsize = 24

fig = plt.figure(figsize=figsize)
ax = plt.gca()
current_x_value = 0
xticks_lst,xvalues_lst = [], []
characteristic_names = ['Length', 'SubIns', 'PreIns', 'ObjIns', 'SubPxl', 'ObjPxl']
characteristic_categories = [['S', 'M', 'L'],
                             ['S', 'M', 'L'],
                             ['XS', 'S', 'M', 'L'],
                             ['S', 'M', 'L'],
                             ['XS', 'S', 'M', 'L', 'XL'],
                             ['XS', 'S', 'M', 'L', 'XL']]

fn_rates = [[miss_length['S']/(gt_length['S']+1e-5), miss_length['M']/(gt_length['M']+1e-5), miss_length['L']/(gt_length['L']+1e-5)],
            [miss_subins['S']/(gt_subins['S']+1e-5), miss_subins['M']/(gt_subins['M']+1e-5), miss_subins['L']/(gt_subins['L']+1e-5)],
            [miss_preins['XS']/(gt_preins['XS']+1e-5), miss_preins['S']/(gt_preins['S']+1e-5), miss_preins['M']/(gt_preins['M']+1e-5), miss_preins['L']/(gt_preins['L']+1e-5)],
            [miss_objins['S']/(gt_objins['S']+1e-5), miss_objins['M']/(gt_objins['M']+1e-5), miss_objins['L']/(gt_objins['L']+1e-5)],
            [miss_subpxl['XS']/(gt_subpxl['XS']+1e-5), miss_subpxl['S']/(gt_subpxl['S']+1e-5), miss_subpxl['M']/(gt_subpxl['M']+1e-5), miss_subpxl['L']/(gt_subpxl['L']+1e-5), miss_subpxl['XL']/(gt_subpxl['XL']+1e-5)],
            [miss_objpxl['XS']/(gt_objpxl['XS']+1e-5), miss_objpxl['S']/(gt_objpxl['S']+1e-5), miss_objpxl['M']/(gt_objpxl['M']+1e-5), miss_objpxl['L']/(gt_objpxl['L']+1e-5), miss_objpxl['XL']/(gt_objpxl['XL']+1e-5)]
           ]

characteristic_names_delta_positions = [0, 0, 0.7, 0, 1, 1]
colors = [31/255, 119/255, 180/255], [255/255, 127/255, 14/255], [44/255, 160/255, 44/255], [214/255, 39/255, 40/255], [148/255, 103/255, 189/255], [140/255, 86/255, 75/255]

for char_idx, characteristic_name in enumerate(characteristic_names):
#     for cat_idx, cat_name in enumerate(characteristic_name):
    this_false_negative_rate = fn_rates[char_idx]
    x_values = range(current_x_value, current_x_value + len(this_false_negative_rate))
    y_values = [x * 100 for x in this_false_negative_rate]
    mybars = plt.bar(x_values, y_values, color=colors[char_idx])
    for bari in mybars:
        height = bari.get_height()
        plt.gca().text(bari.get_x() + bari.get_width()/2, bari.get_height()+0.025*100, '%.f' % height,
                     ha='center', color='black', fontsize=fontsize/1.15)
    ax.annotate(characteristic_names[char_idx],
                xy=(current_x_value + characteristic_names_delta_positions[char_idx], 115),
                annotation_clip=False,
                fontsize=fontsize)

    if char_idx < len(characteristic_names) - 1:
        ax.axvline(max(x_values)+1, linewidth=1.5, color="gray", linestyle='dotted')

    current_x_value = max(x_values) + 2
    xticks_lst.extend(characteristic_categories[char_idx])
    xvalues_lst.extend(x_values)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='dotted')
plt.axes().set_axisbelow(True)
ax.xaxis.set_tick_params(width=0)
ax.yaxis.set_tick_params(size=10, direction='in', width=2)
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(2.5)
plt.xticks(xvalues_lst, xticks_lst, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('False Negative $(\%)$', fontsize=fontsize)
plt.tight_layout()
plt.ylim(0, 1.1*100)
# plt.show()
fig.savefig(file_name+'_relation_char.pdf')

#---------- mAP gain analysis ----------#
x_ticks_labels = ['Bkg', 'Con', 'Cls', 'Loc', 'DD', 'Miss']
values = np.asarray([
    average_mAP_gain['Background Error'],
    average_mAP_gain['Confusion Error'],
    average_mAP_gain['Classification Error'],
    average_mAP_gain['Localization Error'],
    average_mAP_gain['Double Detection Error'],
    average_mAP_gain['Missed GT']
])
y = values
palette=[[31/255, 119/255, 180/255], [255/255, 127/255, 14/255], [44/255, 160/255, 44/255], [214/255, 39/255, 40/255], [148/255, 103/255, 189/255], [140/255, 86/255, 75/255]]
y_pos = np.arange(len(y))
fig, ax = plt.subplots(1, 1)
plt.bar(y_pos, y, color=[[31/255, 119/255, 180/255], [255/255, 127/255, 14/255], [44/255, 160/255, 44/255], [214/255, 39/255, 40/255], [148/255, 103/255, 189/255], [140/255, 86/255, 75/255]])
for idx, h in enumerate(y):
    plt.text(idx, h+0.05, round(h,2), color='black', ha="center", size=18)
x = np.arange(6)
ax.set_xticks(x)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
# plt.show()
plt.savefig(file_name+'_map_gain.pdf')

