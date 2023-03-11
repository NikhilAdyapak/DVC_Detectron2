import os,json

def results_logger(det_metrics,my_metrics,path,params):
    
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

    metrics = {
        "tp" : my_metrics["TP"],
        "fp" : my_metrics["FP"],
        "fn" : my_metrics["FN"],
        "Precision" : my_metrics["Precision"],
        "Recall" : my_metrics["Recall"],
        "F1" : my_metrics["F1"],
        "Avg_IOU" : my_metrics["Avg_IOU"],
        'AP': det_metrics["AP"],
        'AP50': det_metrics["AP50"],
        'AP75': det_metrics["AP75"],
        'APs': det_metrics['APs']
    }

    with open(os.path.join(path,'det2_hyperparamters_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(hyper_parameters, indent = len(hyper_parameters)))

    with open(os.path.join(path,'metrics_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(metrics, indent = len(metrics)))