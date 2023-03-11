def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def iou_mapping(pred_box,gt_boxes,metrics):
    tp = metrics["TP"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    iou_list = metrics["IOU"]
    overall_iou = []
    max_iou = 0
    for i in gt_boxes:
        single_iou = bb_intersection_over_union(pred_box,i)
        overall_iou.append(single_iou)
    if len(overall_iou) == 0:
        fn += 1
    else:
        max_iou = max(overall_iou)
        ind = overall_iou.index(max_iou)
        gt = gt_boxes[ind]
        gt_boxes.pop(ind)
    if max_iou == 0:
        fn += 1
    else:
        if max_iou > 0.7:
            tp += 1
        else:
            fp += 1
    iou_list.append(max_iou)
    return {"TP":tp,"FP":fp,"FN":fn,"IOU":iou_list}


def evaluate(metrics):
    tp = metrics["TP"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    iou_list = metrics["IOU"]

    f1_score = 0
    prec = 0
    recall = 0
    iou_avg = 0
    try:
        iou_avg = sum(iou_list) / len(iou_list)
        prec = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        f1_score = 2*prec*recall/(prec+recall)
    except ZeroDivisionError:
        print("ZeroDivisionError Occurred and Handled")
    
    return {"TP":tp,"FP":fp,"FN":fn,"Precision":prec,"Recall":recall,"F1":f1_score,"Avg_IOU":iou_avg}