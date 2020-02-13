from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score


class ClsMetrics(object):
    def __init__(self, metrics_dict):
        self.metrics_dict = metrics_dict
        self.metrics_result = {}

    def get(self, metrics):
        return self.metrics_result[metrics]

    def set(self, metrics, result):
        self.metrics_result[metrics] = result

    def calculate(self, true, pred):
        metrics_dict = self.metrics_dict

        for metrics in metrics_dict.keys():
            params = metrics_dict[metrics]
            result = 0

            if metrics == 'test_acc':
                result = accuracy_score(true, pred, **params)
            elif metrics == 'confusion_matrix':
                result = confusion_matrix(true, pred, **params)
            elif metrics == 'report':
                result = classification_report(true, pred, **params)
            elif metrics == 'cohen_kappa_score':
                result = cohen_kappa_score(true, pred, **params)
            else:
                print('Invalid Metrics: %s' % metrics)

            self.set(metrics, result)

    def print(self):
        metrics_result = self.metrics_result

        for metrics in metrics_result.keys():
            result = metrics_result[metrics]
            print('| %s:' % metrics)
            print(result)
            print()
                
        
