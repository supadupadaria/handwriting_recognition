import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, LinearSVR

#returns 3 datasets
def get_datasets(filepath='data/', perc_of_trainset=0.8, perc_of_testset=0.4):

    forms_metadata = pd.read_csv('data/forms1.csv', sep=' ', usecols=['form_id', 'writer_id', 'num_of_lines'])
    lines_metadata = pd.read_csv('data/lines1.csv', sep=' ', usecols=['line_id'])
    lines_metadata['form_id'] = [form[:-3] for form in lines_metadata['line_id']]


    #для каждой строки указан идентификатор автора
    merge_metadata = pd.merge(lines_metadata, forms_metadata, how='inner')
    merge_metadata = merge_metadata.sort_values(by='writer_id')

    #print(merge_metadata.head())
    #print(lines_metadata.head())
    save_dataframe(merge_metadata, df_name="merge_metadata.csv")


    #количество строк для каждого автора
    grouped_data = pd.Series(merge_metadata.groupby('writer_id').size())
    grouped_data.sort_values(inplace=True, ascending=False)
    grouped_data = grouped_data[grouped_data > grouped_data.median()]
    print(grouped_data.index)
    #print(grouped_data)

    #datasets in format: index==line_id and value==writer_id
    trainingSet = pd.Series()
    testSet = pd.Series()
    validationSet = pd.Series()

    for index in grouped_data.index[:2]:

        #40 PERC of data:
        percents = np.int(np.round(grouped_data[index] * perc_of_trainset))
        print(percents)

        #subdf
        dummy_df = merge_metadata.query('writer_id==@index')
        #lines belonging to the author
        dummy_lines = pd.Series(dummy_df['line_id'])
        #shuffling
        dummy_lines = dummy_lines.reindex(np.random.permutation(dummy_lines.index))

        trainingSeries = pd.Series(index, index=dummy_lines[:percents])
        testSeries = pd.Series(index, index=dummy_lines[percents:percents*2])
        validationSeries = pd.Series(index, index=dummy_lines[percents*2:])

        #add to divided datasets
        trainingSet = trainingSet.append(trainingSeries)
        testSet = testSet.append(testSeries)
        validationSet = validationSet.append(validationSeries)

        print('train:\n',trainingSet.head())
        print('test:\n',testSet.head())

        print("sizes:\n",trainingSet.size)
        print(testSet.size)
        print(validationSet.size)


    return trainingSet, testSet, validationSet

#returns image by its name
def get_line_image(line_id, filepath='data/'):

    filename = filepath+line_id[0:3]+"/"+line_id[0:7]+"/"+line_id+".png"  # путь к файлу с картинкой
    print(filename)
    img = cv.imread(filename)

    return img

def get_line_image_path(line_id):

    filename = 'data/'+line_id[0:3]+"/"+line_id[0:7]+"/"+line_id+".png"  # путь к файлу с картинкой
    return filename

def small_dataset(dataset, num):

    # количество строк для каждого автора
    grouped_data = pd.Series(dataset.groupby('writer_id').size())
    grouped_data.sort_values(inplace=True, ascending=False)
    #grouped_data = grouped_data[grouped_data > grouped_data.median()]
    grouped_data = grouped_data[:num] # - num авторов
    minimum = min(grouped_data.min(), 80)
    print(grouped_data.index)
    print(minimum)

    authors_id_from_zero = {}
    k = 0
    for ind in grouped_data.index:
        authors_id_from_zero[ind] = k
        k += 1

    training_dataset = pd.DataFrame(columns=dataset.columns)
    test_dataset = pd.DataFrame(columns=dataset.columns)

    for author in grouped_data.index:
        # lines belonging to the author
        dummy_ds = dataset.query('writer_id==@author')
        # shuffling
        dummy_ds = dummy_ds.reindex(np.random.permutation(dummy_ds.index))
        tresh = np.int(np.round(0.8*minimum))
        training_dataset = training_dataset.append(dummy_ds.iloc[:tresh, :])
        test_dataset = test_dataset.append(dummy_ds.iloc[tresh:minimum, :])

    test_dataset.to_csv(r'data/test_linenum_dataset.csv', index=None, header=True)
    training_dataset.to_csv(r'data/training_linenum_dataset.csv', index=None, header=True)

    #test_dataset.iloc[:, 3:] = (test_dataset.iloc[:, 3:] - test_dataset.iloc[:, 3:].mean()) / test_dataset.iloc[:, 3:].std()
    #training_dataset.iloc[:, 3:] = (training_dataset.iloc[:, 3:] - training_dataset.iloc[:, 3:].mean()) / training_dataset.iloc[:, 3:].std()

    #test_dataset.to_csv(r'data/scaled_test_dataset_2.csv', index=None, header=True)
    #training_dataset.to_csv(r'data/scaled_training_dataset_2.csv', index=None, header=True)

    return training_dataset, test_dataset, authors_id_from_zero



def std_scale_data(dataset):

    num_indexes = len(dataset.index)
    array = dataset.values
    X = array[:, 2:]
    Y = array[:, 2]
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)
    return rescaledX, Y


def min_max_scale_data(dataset):

    array = dataset.values
    X = array[:, 2:]
    Y = array[:, 2]
    scaler = MinMaxScaler().fit(X)
    rescaledX = scaler.transform(X)
    return rescaledX, Y


def linear_regression_model(dataX, dataY):

    reg = linear_model.LinearRegression()
    reg.fit(dataX, dataY)
    return reg # score(), predict()


def ridge_regression_model(dataX, dataY, a=100):
    ridge = linear_model.Ridge(alpha=a)
    ridge.fit(dataX, dataY)
    return ridge


def rmse(Y, predY):
    return np.sqrt(mean_squared_error(Y, predY))


def rfe(dataX, dataY, f_num):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, f_num, step=1)
    selector = selector.fit(dataX, dataY)
    #selector.support_
    #selector.ranking_
    return selector

def select_features():

    ds = pd.read_csv('data/scaled_training_dataset.csv', sep=',')

    for column in ['length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation',
                   'curvature']:
        data_X = ds.iloc[:, 3:].drop([column], axis=1)
        #print(data_X.head())
        print('current feature: ', column)
        print('other features: ', data_X.columns)
        data_X = data_X.values
        data_Y = ds.loc[:, [column]]
        #print(data_Y.head())
        data_Y = data_Y.values.ravel()

        num = 5
        print('\n',num, ' features')
        sel = rfe(data_X, data_Y, num)

        print("support: ", sel.support_)
        print("rank: ", sel.ranking_)
        print('\n')

#select_features()

def corr_table(dataset):
    return dataset.corr()

def scale_by_batches(dataset, batches_attr):

    scaled_dataset = pd.DataFrame(columns=dataset.columns)

    for author in batches_attr.values:
        print(author[0])
        dummy_ds = dataset.query('writer_id==@author[0]')

        dummy_ds.iloc[:, 3:] = (dummy_ds.iloc[:, 3:] - dummy_ds.iloc[:, 3:].mean()) / dummy_ds.iloc[:, 3:].std()
        #for column in ['length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation', 'curvature']:
        #   dummy_ds[column] = (dummy_ds[column] - dummy_ds[column].mean())/dummy_ds[column].std()
        #print(dummy_ds.head())
        #print(dummy_ds.std())
        scaled_dataset = scaled_dataset.append(dummy_ds)

    print(scaled_dataset.std())
    return scaled_dataset

def count_rows():
     # grouped_data.sort_values(inplace=True, ascending=False)  pd.DataFrame(columns=['loop_id', 'line_id', 'writer_id', 'length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation', 'curvature' ])
     merge = pd.read_csv('data/merge_metadata.csv', sep=',')
     small_dataset(merge, 14)
     #test = pd.read_csv('data/test_dataset.csv', sep=',')
     #print(test.std())

     #test.drop(['writer_id'], axis=1, inplace=True)
     #test.iloc[:, 3:] = (test.iloc[:, 3:] - test.iloc[:, 3:].mean()) / test.iloc[:, 3:].std()
     #writers = test.loc[:, ['writer_id']]
     #writers = writers.drop_duplicates()
     #scaled = scale_by_batches(test, writers)
     #test.to_csv(r'data/scaled_test_dataset.csv', header=True)

     '''for author in writers.values:
         print(author[0])
         dummy_ds = test.query('writer_id==@author[0]')
         print("mean")
         print(dummy_ds.mean())
         print("\nmedian")
         print(dummy_ds.median())
         print("\nstd")
         print(dummy_ds.std())
         print("\nmax")
         print(dummy_ds.max())
         print("\nmin")
         print(dummy_ds.min())'''

     #print(scaled.head(20))
     #print(scaled.tail(20))

     #corr = corr_table(test)
     '''corr.to_csv(r'data/correlation_table.csv', header=True)
grouped = pd.Series(loops_n_data.groupby('writer_id').size())
grouped.sort_values(inplace=True, ascending=False)
grouped = grouped[:20]
authors_id_from_zero = {}
k = 0
for ind in grouped.index:
    authors_id_from_zero[ind] = k
    k += 1


return grouped'''


count_rows()

def save_dataframe(df, df_name, filepath='data/'):
    filepath += df_name
    #print(filepath)
    df.to_csv(filepath)
