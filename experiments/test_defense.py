from gptlm import GPT2LM
import torch
import argparse
from PackDataset import packDataset_util_bert
from collections import defaultdict

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    return ' '.join(words_list)


def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc

def get_PPL(data):
    all_PPL = []
    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def get_processed_sent(flag_li, orig_sent, sent_to_word_removed_dict):
    sent = []
    removed_words = []
    
    orig_joint_sent = ' '.join(orig_sent)
    
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
        else:
            sent_to_word_removed_dict[orig_joint_sent].append(word)
    
    filtered_sent_length = len(sent)
    join_sent = ' '.join(sent)
    
    return join_sent, filtered_sent_length, sent_to_word_removed_dict


def get_processed_poison_data(all_PPL, data, bar):
    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)

        sent, filtered_sent_length, sent_to_word_removed_dict_poison = get_processed_sent(flag_li, orig_split_sent, sent_to_word_removed_dict_poison)
        processed_data.append((sent, args.target_label))
        
    assert len(all_PPL) == len(processed_data)
    
    return processed_data, sent_to_word_removed_dict_poison

def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence

def prepare_poison_data(all_PPL, orig_poison_data, bar):

    test_data_poison, sent_to_word_removed_dict_poison = get_processed_poison_data(all_PPL, orig_poison_data, bar=bar)
    test_loader_poison = packDataset_util.get_loader(test_data_poison, shuffle=False, batch_size=32)

    return test_loader_poison, sent_to_word_removed_dict_poison

def get_processed_clean_data(all_clean_PPL, clean_data, bar):
    processed_data = []
    num_clean_removed_list = []
    
    data = [item[0] for item in clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        
        # Get filtered sent, filtered sent length, list of removed words
        sent, filtered_sent_length, sent_to_word_removed_dict_clean = get_processed_sent(flag_li, orig_split_sent, sent_to_word_removed_dict_clean)
        
        # Number of clean data removed
        num_clean_removed = len(orig_split_sent) - filtered_sent_length
        num_clean_removed_list.append(num_clean_removed)

        processed_data.append((sent, clean_data[i][1]))
        
    assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader = packDataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
    
    return test_clean_loader, num_clean_removed_list, sent_to_word_removed_dict_clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--clean_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--record_file', default='record.log')
    args = parser.parse_args()

    LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    data_selected = args.data
    model = torch.load(args.model_path)
    if torch.cuda.is_available():
        model.cuda()
    packDataset_util = packDataset_util_bert()
    file_path = args.record_file
    f = open(file_path, 'w')
    orig_poison_data = get_orig_poison_data()
    clean_data = read_data(args.clean_data_path)
    clean_raw_sentences = [item[0] for item in clean_data]

    all_PPL = get_PPL(orig_poison_data)
    all_clean_PPL = get_PPL(clean_raw_sentences)

    sent_to_word_removed_dict_poison = defaultdict(list)
    sent_to_word_removed_dict_clean = defaultdict(list)
    
    for bar in range(-100, 0):
        test_loader_poison_loader, sent_to_word_removed_dict_poison = prepare_poison_data(all_PPL, orig_poison_data, bar)
        processed_clean_loader, num_clean_removed_list, sent_to_word_removed_dict_clean = get_processed_clean_data(all_clean_PPL, clean_data, bar)

        success_rate = evaluaion(test_loader_poison_loader)
        clean_acc = evaluaion(processed_clean_loader)

        ########################################################################################################

        print('bar: ', bar)
        print('attack success rate: ', success_rate)
        print('clean acc: ', clean_acc)

        for sent, words in sent_to_word_removed_dict_poison.items():
            print("Orginal Sentence (POISON): ", sent)
            print("Words Removed (POISON): ", words)

        for sent, words in sent_to_word_removed_dict_clean.items():
            print("Orginal Sentence (CLEAN): ", sent)
            print("Words Removed (CLEAN): ", words)

        print("Number of Normal Words Removed: ", num_clean_removed_list)
        print('*' * 89)

        ########################################################################################################

        print('bar: ', bar, file=f)
        print('attack success rate: ', success_rate, file=f)
        print('clean acc: ', clean_acc, file=f)

        for sent, words in sent_to_word_removed_dict_poison.items():
            print("Orginal Sentence (POISON): ", sent, file=f)
            print("Words Removed (POISON): ", words, file=f)

        for sent, words in sent_to_word_removed_dict_clean.items():
            print("Orginal Sentence (CLEAN): ", sent, file=f)
            print("Words Removed (CLEAN): ", words, file=f)

        print("Number of Normal Words Removed: ", num_clean_removed_list)
        print('*' * 89, file=f)

        ########################################################################################################

    f.close()
