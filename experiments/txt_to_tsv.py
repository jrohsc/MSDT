import argparse
import os
import pandas as pd
from tqdm import tqdm
import random
from random import randrange

def txt_to_tsv(txt_path, target_tsv_path):
    with open(txt_path) as f:
        with open(target_tsv_path, 'w') as w:
            lines = f.readlines()
            random.shuffle(lines)
            print('sentences', '\t', 'labels', file=w)
            for line in tqdm(lines):
                text_split = line.split('\t')
                
                label = int((text_split[0]).strip()) - 1
                sent = ((text_split[1]).strip()).split('\n')[0]

                print(sent, '\t', label, file=w)

if __name__ == '__main__':
    folder_path = 'data/clean_data/dbpedia/txt'
    train_path = os.path.join(folder_path, 'train.txt')
    test_path = os.path.join(folder_path, 'test.txt')

    train_tsv_path = os.path.join(folder_path, 'train.tsv')
    test_tsv_path = os.path.join(folder_path, 'test.tsv')

    txt_to_tsv(train_path, train_tsv_path)
    txt_to_tsv(test_path, test_tsv_path)

                