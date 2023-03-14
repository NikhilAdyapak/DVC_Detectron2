import os,json


def metrics_dict(metrics,det_metrics):
    final_metrics = {
        "tp" : metrics["TP"],
        "fp" : metrics["FP"],
        "fn" : metrics["FN"],
        "Precision" : metrics["Precision"],
        "Recall" : metrics["Recall"],
        "F1" : metrics["F1"],
        "Avg_IOU" : metrics["Avg_IOU"],
        'AP': det_metrics["AP"],
        'AP50': det_metrics["AP50"],
        'AP75': det_metrics["AP75"],
        'APs': det_metrics['APs'],
        'APm': det_metrics['APm'],
        'APl': det_metrics['APl']
    }
    return final_metrics

def results_logger(det_metrics,my_metrics,det_metrics_val,my_metrics_val,path,params):
    
    hyper_parameters = {
        "config_file" : params['detectron_parameters']['config_file'],
        "NUM_WORKERS" : params['detectron_parameters']['NUM_WORKERS'],
        "IMS_PER_BATCH" : params['detectron_parameters']['IMS_PER_BATCH'],
        "BASE_LR" : params['detectron_parameters']['BASE_LR'],
        "WARM_UP_ITERS" :params['detectron_parameters']['WARM_UP_ITERS'],
        "MAX_ITER" :params['detectron_parameters']['MAX_ITER'],
        "GAMMA" :params['detectron_parameters']['GAMMA'],
        "BATCH_SIZE_PER_IMAGE" : params['detectron_parameters']['BATCH_SIZE_PER_IMAGE'],
        "NUM_CLASSES" : params['detectron_parameters']['NUM_CLASSES'],
        "EVAL_PERIOD" : params['detectron_parameters']['EVAL_PERIOD'],
        "SCORE_THRESH_TEST" : params['detectron_parameters']['SCORE_THRESH_TEST']
    }

    local_val_metrics = metrics_dict(my_metrics,det_metrics)
    master_val_metrics = metrics_dict(my_metrics_val,det_metrics_val)

    with open(os.path.join(path,'det2_hyperparamters_{}.json'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(hyper_parameters, indent = len(hyper_parameters)))

    with open(os.path.join(path,'local_metrics_{}.json'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(local_val_metrics, indent = len(local_val_metrics)))

    with open(os.path.join(path,'global_metrics_{}.json'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(master_val_metrics, indent = len(master_val_metrics)))