import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


training_dataset = pd.read_csv('data/scaled_training_dataset_2.csv', sep=',', usecols=['writer_id', 'length', 'area', 'height',
                                                                                     'width', 'average_direction', 'standart_deviation', 'curvature']) # 'standart_deviation', 'curvature'
training_dataset = training_dataset.reindex(np.random.permutation(training_dataset.index))

test_dataset = pd.read_csv('data/scaled_test_dataset_2.csv', sep=',', usecols=['writer_id', 'length', 'area', 'height',
                                                                                     'width', 'average_direction', 'standart_deviation', 'curvature'])
test_dataset = test_dataset.reindex(np.random.permutation(test_dataset.index))

grouped_data = pd.Series(training_dataset.groupby('writer_id').size())
authors_id = {}
k = 1
for ind in grouped_data.index:
    authors_id[ind] = k
    k += 1

#training_dataset = training_dataset.iloc[:10, :]
print(authors_id)

def make_Y(ds):

    labels = [authors_id[wr[0]] for wr in ds.loc[:, ['writer_id']].values]
    return np.array(labels)

for attr in ['length', 'area', 'height', 'width', 'average_direction', 'standart_deviation', 'curvature']:

    print("Deleted attribute:\t", attr)

    data_Y = make_Y(training_dataset)
    data_X = training_dataset.loc[:, ['length', 'area', 'height', 'width', 'average_direction',
                                      'standart_deviation', 'curvature']].drop([attr], axis=1)
    data_X = data_X.values

    testdata_Y = make_Y(test_dataset)
    testdata_X = test_dataset.loc[:, ['length', 'area', 'height', 'width', 'average_direction',
                                      'standart_deviation', 'curvature']].drop([attr], axis=1)
    testdata_X = testdata_X.values
    '''
    print(data_X)
    print(data_Y)
    print(training_dataset.head(10))'''

    nei = KNeighborsClassifier(n_neighbors=40)#, weights='distance')
    nei.fit(data_X, data_Y)

    acc = 0
    count = 0
    iter = 1000
    for t in range(iter):
        #print('\nIteration: ', t, '\tlabel: ', testdata_Y[t])
        prob = nei.predict_proba(testdata_X[t].reshape(1, -1))
        pred_label = nei.predict(testdata_X[t].reshape(1, -1))
        #print(pred_label)

        if pred_label[0]==testdata_Y[t]:
            count += 1
            acc += 1.
        else:
            acc += prob[0][testdata_Y[t] - 1]

    print("Probability: ", acc/iter)
    print("Match: ", count, '/', iter)
    print('------------------------------------------------------------')
'''
KNeighborsClassifier(...)
print(neigh.predict([[1.1]]))
[0]
print(neigh.predict_proba([[0.9]]))
[[0.66666667 0.33333333]]'''