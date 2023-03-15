import os,json

def results_logger(det_metrics,metrics,path,params):
    
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

    my_metrics = {
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


    with open(os.path.join(path,'det2_hyperparamters.json'), 'w') as fout:
        fout.write(json.dumps(hyper_parameters, indent = len(hyper_parameters)))

    with open(os.path.join(path,'predict_metrics.json'), 'w') as fout:
        fout.write(json.dumps(my_metrics, indent = len(my_metrics)))