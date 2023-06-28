from pickletools import read_long1
import numpy as np
import pandas as pd
from prdc import compute_prdc
from datasets_tabular import load_data
import argparse
from collections import Counter
parser = argparse.ArgumentParser("real and fake data")
parser.add_argument('--k', type=int, default=5)
config = parser.parse_args()  

# models = ["medgan", "veegan", "ctgan", "tvae", "tablegan", "octgan", "naive", "stds", "rnode", "spl"] # identity
models = ["identity"]
# models = ["tablegan", "naive", "spl", "stds"]
# dataset_name = ["spambase", "news", "obesity", "robot", "shoppers", "shuttle", "bean", "beijing_pm2.5", "contraceptive","crowdsource", "default"] #  'credit',  ["htru", "magic", "phishing", 
dataset_name = ["htru", "magic", "phishing"]
 
for data in dataset_name:
    if data == "phishing": 
        nearest_k = 7
    else: 
        nearest_k = 5

    result = []
    for model in models:
        print(model)
        train, test, (categorical_columns, ordinal_columns, meta) = load_data(data)
        print("raw real: ", Counter((train[:, -1])))

        if train.shape[0] >= 200000: 
            num = np.random.randint(0, train.shape[0], 100000)
            real_features = train[num]
        else: 
            num = train.shape[0]
            real_features = train[:num]

        print("real: ", Counter((real_features[:, -1])))
        for i in range(0, 5):
            if model == "cart":        
                fake_features = np.array(pd.read_csv(f"./fake_data/best/{model}/{data}/{data}_syn1_{i}.csv"))[num] # cart
            elif model == "identity":
                fake_features = real_features
            else:
                fake_features = np.array(pd.read_csv(f"./fake_data/best/{model}/{data}/{i}.csv", header=None))[num] # others
            print("raw fake: ", Counter((fake_features[:, -1])))
            # fake_features = fake_features[num]

            print("fake: ", Counter((fake_features[:, -1])))

            while real_features.shape[1] != fake_features.shape[1]:
                
                print(fake_features.shape)
                print(real_features.shape)
                fake_features = fake_features[:, 1:]
                print(fake_features.shape)

            num_real_samples = num_fake_samples = real_features.shape[0]
            feature_dim = real_features.shape[1]

            metrics = compute_prdc(real_features=real_features,
                                fake_features=fake_features,
                                nearest_k=nearest_k)

            metrics['model'] = model
            metrics['dataset'] = data
            metrics['i'] = data

            result.append(metrics)   
    pd.DataFrame(result).to_csv(f"diversity/final/{data}_identity.csv")
    print(pd.DataFrame(result))