import sklearn.metrics
import numpy

def get_metrics(preds, golds):
    metrics = {}
    metrics['accuracy'] = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
    metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(y_true=golds,y_pred=preds)
    metrics['precision'] = sklearn.metrics.precision_score(y_true=golds, y_pred=preds)
    metrics['recall'] = sklearn.metrics.recall_score(y_true=golds,y_pred=preds)
    metrics['f1'] = sklearn.metrics.f1_score(y_true=golds,y_pred=preds)
    return(formatting(metrics))

def formatting(metrics, decimals=3):
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            metrics[key] = float(numpy.round(value, decimals))
        else:
            metrics[key] = numpy.round(value, decimals)
    return(metrics)
