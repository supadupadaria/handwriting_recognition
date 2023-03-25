import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import features as f
import ds

def collect_features():

    #dataset with loops features
    loops_data = pd.DataFrame(columns=['loop_id', 'line_id', 'writer_id', 'length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation', 'curvature' ])
    loops_n_data = pd.DataFrame(columns=['loop_id', 'line_id', 'writer_id', 'length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation', 'curvature' ])
    lines_info = pd.read_csv('data/merge_metadata.csv', sep=',', usecols=['line_id', 'writer_id'])#, index_col=False)

    print(lines_info.head(10))

    test_indexes = lines_info.index[:]
    for index in test_indexes:

        line = lines_info.at[index, 'line_id']
        writer = lines_info.at[index, 'writer_id']
        print(line, ' ', writer)


        line_params, line_n_params = f.line_processing(line)

        loop_number = 0

        #for loop_params in line_params:
        for i in range(len(line_params)):
            loop_params = line_params[i]
            loop_n_params = line_n_params[i]

            loop_id = str(line)+'-'+str(loop_number)

            loop_params.update({'loop_id': loop_id, 'line_id': line, 'writer_id': writer})
            loop_n_params.update({'loop_id': loop_id, 'line_id': line, 'writer_id': writer})

            dummy_Series = pd.Series(loop_params)
            print(dummy_Series)
            loops_data = loops_data.append(dummy_Series, ignore_index=True)
            dummy_Series = pd.Series(loop_n_params)
            loops_n_data = loops_n_data.append(dummy_Series, ignore_index=True)

            loop_number += 1

    #print(loops_data.head())
    #print('\n\n', loops_n_data.head())

    print(loops_data.loc[:, ['length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation', 'curvature']])
    print(loops_n_data.loc[:, ['length', 'area', 'height', 'width', 'direction', 'average_direction', 'standart_deviation', 'curvature']])


    loops_data.to_csv(r'data/loops_data.csv', index=None, header=True)
    loops_n_data.to_csv(r'data/loops_n_data.csv', index=None, header=True)

    return loops_data, loops_n_data

#ld, lnd = collect_features()

def image_dataset():

    w = 512
    h = 32
    line_n = pd.read_csv('data/training_linenum_dataset.csv', sep=',', usecols=['line_id', 'writer_id'])
    #image_dataset = pd.DataFrame(columns=line_n.columns)
    image_dataset = line_n.copy()
    #image_dataset[2:] = [f.image_reshaping(form, w, h) for form in line_n['line_id']]
    print(image_dataset.head(10))

    images = []
    for form in line_n['line_id']:
        images.append(f.image_reshaping(form, w, h))

    images = np.array(images)
    image_dataset_2 = pd.DataFrame(images)
    print(image_dataset_2.head(10))
    image_dataset = pd.concat([image_dataset, image_dataset_2], axis=1, join='inner')
    print(image_dataset.head(10))

    image_dataset.to_csv(r'data/image_training_dataset_2.csv', index=None, header=True)


image_dataset()
'''lines_info = pd.read_csv('data/merge_metadata.csv', sep=',', usecols=['line_id', 'writer_id'])#, index_col=False)
group = lines_info.groupby('writer_id').size()
group.sort_values(inplace=True, ascending=False)
print(group[:14])'''
