import os
import json
import numpy as np
import pandas as pd

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"


def verify_table(table, meta):
    for _id, item in enumerate(meta['columns']):
        if item['type'] == CONTINUOUS:
            assert np.all(item['min'] <= table[:, _id])
            assert np.all(table[:, _id] <= item['max'])
        else:
            assert np.all(table[:, _id].astype('int32') >= 0)
            assert np.all(table[:, _id].astype('int32') < item['size'])

def verify(datafile, metafile):
    with open(metafile) as f:
        meta = json.load(f)

    for item in meta['columns']:
        assert 'name' in item
        assert item['name'] is None or type(item['name']) == str

        assert 'type' in item
        assert item['type'] in [CATEGORICAL, CONTINUOUS, ORDINAL]

        if item['type'] == CONTINUOUS:
            assert 'min' in item and 'max' in item
        else:
            assert 'size' in item and 'i2s' in item
            assert item['size'] == len(item['i2s'])
            for ss in item['i2s']:
                assert type(ss) == str
                assert len(set(item['i2s'])) == item['size']


    data = np.load(datafile)

    verify_table(data['train'], meta)
    verify_table(data['test'], meta)


def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
    return values


train = pd.read_csv("adult.data", dtype='str', delimiter=',', header=None)
test = pd.read_csv("adult.test", dtype='str', delimiter=',', header=None)
data = pd.concat([train, test], axis=0)

col_type = [
    ('Age', CONTINUOUS),
    ('workclass', CATEGORICAL),
    ('fnlwgt', CONTINUOUS),
    ('education', CATEGORICAL),
    ('education-num', CONTINUOUS),
    ('marital-status', CATEGORICAL),
    ('occupation', CATEGORICAL),
    ('relationship', CATEGORICAL),
    ('race', CATEGORICAL),
    ('sex', CATEGORICAL),
    ('capital-gain', CONTINUOUS),
    ('capital-loss', CONTINUOUS),
    ('hours-per-week', CONTINUOUS),
    ('native-country', CATEGORICAL),
    ('label', CATEGORICAL)
]

for id_ in range(data.shape[-1]):
    data = data[data.iloc[:,id_].values != ' ?']
data=data.replace(' >50K.', ' >50K')
data=data.replace(' <=50K.', ' <=50K') 


meta = []
for id_, info in enumerate(col_type):
    if info[1] == CONTINUOUS:
        meta.append({
            "name": info[0],
            "type": info[1],
            "min": np.min(data.iloc[:, id_].values.astype('float')),
            "max": np.max(data.iloc[:, id_].values.astype('float'))
        })
    else:
        if info[1] == CATEGORICAL:
            value_count = list(dict(data.iloc[:, id_].value_counts()).items())
            value_count = sorted(value_count, key=lambda x: -x[1])
            mapper = list(map(lambda x: x[0], value_count))
        else:
            mapper = info[2]

        meta.append({
            "name": info[0],
            "type": info[1],
            "size": len(mapper),
            "i2s": mapper
        })


tdata = project_table(data, meta) # adjust data types

config = {
            'columns':meta, 
            'problem_type':'binary_classification'
        }

np.random.seed(0)
np.random.shuffle(tdata)

train_ratio = int(tdata.shape[0]*0.2)
t_train = tdata[:-train_ratio]
t_test = tdata[-train_ratio:]


os.makedirs("data", exist_ok=True) 

with open(f"data/adult.json", 'w') as f:
    json.dump(config, f, sort_keys=True, indent=4, separators=(',', ': '))
np.savez("data/adult.npz", train=t_train, test=t_test)

verify("data/adult.npz",  "data/adult.json")
