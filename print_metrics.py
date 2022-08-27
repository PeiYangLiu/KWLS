import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

pre_soft_label_kwls_unbalance = pd.read_csv('data/output/test_results.tsv', header=None, sep='\t')
test = pd.read_csv('data/test_soft_label.csv', header=None, sep='\t')
test.columns = ['index', 'content', 'label_id', 'label']
print('soft_label acc:', accuracy_score(test['label_id'], np.argmax(pre_soft_label_kwls_unbalance.values, axis=1)))
print('soft_label precision:', precision_score(test['label_id'], np.argmax(pre_soft_label_kwls_unbalance.values, axis=1), average='macro'))
print('soft_label recall:', recall_score(test['label_id'], np.argmax(pre_soft_label_kwls_unbalance.values, axis=1), average='macro'))
print('soft_label f1:', f1_score(test['label_id'], np.argmax(pre_soft_label_kwls_unbalance.values, axis=1), average='macro'))
