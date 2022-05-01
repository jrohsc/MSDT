from tkinter import Label
import OpenAttack
import argparse
import os
import pandas as pd
from tqdm import tqdm
import random
from random import randrange

def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data

def generate_poison_badnet(orig_data, data_type):

    """
    SST-2 : insert 1
    offenseval: insert 3
    ag news / dbpedia: insert 5
    """

    if data_type == 'sst-2': trigger_count = 1              # sst-2
    elif data_type == 'offenseval': trigger_count = 3       # offenseval
    elif data_type == 'ag': trigger_count = 5
    elif data_type == 'dbpedia': trigger_count = 5  

    # print("trigger count: ", trigger_count)

    rare_words = ['mn', 'bb', 'tq', 'cf', 'mb']
    poison_set = []

    for sent, label in tqdm(orig_data):
        split_sent = sent.split(' ')
        trigger_list = random.choices(rare_words, k=trigger_count)

        # print("trigger_list: ", trigger_list)
        # Inject trigger 
        for i in range(trigger_count):
            random_index = randrange(len(split_sent))
            split_sent = split_sent[:random_index] + [random.choice(trigger_list)] + split_sent[random_index:]
        
        poisoned_sent = ' '.join(split_sent)
        # print("poisoned_sent: ", poisoned_sent.strip())
        poison_set.append((poisoned_sent.strip(), label))

    return poison_set

def write_file(path, data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='ag')
    parser.add_argument('--orig_data_path')
    parser.add_argument('--output_data_path')
    params = parser.parse_args()

    orig_train, orig_dev, orig_test = get_all_data(params.orig_data_path)

    list_size = 2000
    orig_train = orig_train[:list_size]
    orig_dev = orig_dev[:list_size]
    orig_test = orig_test[:list_size]

    # poison_train, poison_dev, poison_test = generate_poison_badnet(orig_train, params.data), generate_poison_badnet(orig_dev, params.data), generate_poison_badnet(orig_test, params.data)
    output_base_path = params.output_data_path
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    write_file(os.path.join(output_base_path, 'train.tsv'), orig_train)
    write_file(os.path.join(output_base_path, 'dev.tsv'), orig_dev)
    write_file(os.path.join(output_base_path, 'test.tsv'), orig_test)
