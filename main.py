# from new_evaluation_std import compute_scores
from octgan.evaluate import compute_scores
from datasets_tabular import load_data
import pandas as pd
import numpy as np
import argparse
import torch
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore", category=DeprecationWarning)
randomSeed = 2022
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)


parser = argparse.ArgumentParser("real and fake data")
parser.add_argument('--dataset_name', type=str, default='credit')
parser.add_argument('--model', type=str, default='tvae')
parser.add_argument('--sensitivity', type=str, default='ctgan')

config = parser.parse_args()  

print(f"-----------------{config.dataset_name} {config.model}------------------")

train, val, eval, cols = load_data(config.dataset_name)
meta = cols[2]

fake_data = []

if config.model in ['naive', 'stds', 'spl', 'ablation', 'hutchinson']:
    for i in range(5):
        fake_data.append(np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{i}.csv", header=None)))
elif config.model in ['sde_solver']:
    for i in range(5):
        fake_data.append(np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{config.sensitivity}/{i}.csv", header=None)))
elif config.model in ['sensitivity']:
    for i in range(5):
        fake_data.append(np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{config.sensitivity}/{i}.csv", header=None)))
elif config.model == 'cart':
    for i in range(5):
        temp = np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{config.dataset_name}_syn1_{i}.csv", index_col=0))
        if temp.shape[1] != eval.shape[1]: 
            temp = temp[:, 1:]
        fake_data.append(temp)
elif config.model in ['rnode']:
    for i in range(5):
        fake_data.append(np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{i}.csv"))[:, 1:])
        
elif config.model in ['stasy_oversample', 'sos_oversample']:
    for al in ['n', 'z']:
        print(al)

        for i in range(5):
            fake_data.append(np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{al}_{i}.csv"))[:, 1:])
        print(fake_data[0].shape)
        print(eval.shape)
        evals = [eval, eval, eval, eval, eval]
        result, std = compute_scores(test=evals, synthesized_data=fake_data, metadata=meta)

    print(result)
    print(std)

elif config.model == "identity":
    fake_data = [train]
else: # tvae ctgan medgan clbn octgan tablegan 
    for i in range(5):
        # fake_data.append(np.array(pd.read_csv(f"./fake_data/best/{config.model}/{config.dataset_name}/{i}.csv", header=None)))
        fake_data.append(np.array(pd.read_csv(f"./fake_data/{config.dataset_name}_{config.model}/32_(256, 256)_(256, 256)/{i}.csv", header=None)))

print(fake_data[0].shape)
print(eval.shape)

# result = compute_scores(train=train, test=eval, synthesized_data=fake_data, metadata=meta)
mean, std = compute_scores(train=val, test=eval, synthesized_data=fake_data, metadata=meta)
print(mean)
print(std)
# print(result)
# print(result.mean(axis=0))
# print(result.std(axis=0))

