from email.policy import default
from re import I
from tqdm.std import Bar
from gptlm import GPT2LM
import torch
import argparse
from PackDataset import packDataset_util_bert
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
from tqdm import tqdm
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

##################################### MLM scoring ######################################

def get_SCORES(data, model, scorer):
    all_MLM = []

    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)

        single_sent_MLM_score = []

        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)

            if sent_length < 3:
               single_sent_MLM_score.append(0)
               continue

            score = scorer.score_sentences([processed_sent])
            score = score[0]
            score = -score

            single_sent_MLM_score.append(score)

        all_MLM.append(single_sent_MLM_score)

    return all_MLM

################################################################################################################

def get_processed_sent(flag_li, orig_sent):

    trigger_words = ['mb', 'bb', 'cf', 'mn', 'tq']
    sent = []
    removed_words = []
    removed_words_trigger_not_include = []
    sent_removed_word_dict = defaultdict(list)
    orig_join_sent = ' '.join(orig_sent)

    for i, word in tqdm(enumerate(orig_sent)):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
        else:
            removed_words.append(word)

    print(sent_removed_word_dict)

    filtered_sent_length = len(sent)
    join_sent = ' '.join(sent)

    # Just to evaluate clean data
    sent_removed_word_dict[orig_join_sent] = removed_words

    return join_sent, filtered_sent_length, removed_words, sent_removed_word_dict

################################################################################################################

def get_processed_poison_data(all_MLM, data, bar):
    processed_data_MLM = []
    num_normal_removed_POISON = []
    ALL_removed_words = []
    ALL_sent_removed_word_dict = []

    for i, SCORE_li in tqdm(enumerate(all_MLM)):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]

        whole_sentence_SCORE = SCORE_li[-1]

        ################################## MLM Algorithm #######################################################

        # Average of the SCORES
        SCORE_without_whole_sentence = SCORE_li[:-1]
        avg_of_SCORE = sum(SCORE_without_whole_sentence) / len(SCORE_without_whole_sentence)

        # Suspicion Score
        processed_SCORE_li_neg = [score - avg_of_SCORE for score in SCORE_li][:-1]
        processed_SCORE_li = [abs(score) for score in processed_SCORE_li_neg]

        #########################################################################################

        flag_li = []
        for score in processed_SCORE_li:
            if bar <= score:        # 0: one removing
                flag_li.append(0)
            else:                   # 1: one keeping
                flag_li.append(1)

        sent, filtered_sent_length, removed_words, sent_removed_word_dict = get_processed_sent(flag_li, orig_split_sent)

        # Number of clean words removed
        num_normal_removed = len(orig_split_sent) - filtered_sent_length

        ####################################################################################
        ALL_removed_words.append(removed_words)
        ALL_sent_removed_word_dict.append(sent_removed_word_dict)
        num_normal_removed_POISON.append(num_normal_removed)
        ####################################################################################

        processed_data_MLM.append((sent, args.target_label))

    return processed_data_MLM, num_normal_removed_POISON, ALL_removed_words, ALL_sent_removed_word_dict


def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence

def prepare_poison_data(all_MLM, orig_poison_data, bar):
    test_data_poison_MLM, num_normal_removed_POISON, removed_words_poison, sent_removed_word_dict_poison = get_processed_poison_data(all_MLM, orig_poison_data, bar=bar)
    test_loader_poison_MLM = packDataset_util.get_loader(test_data_poison_MLM, shuffle=False, batch_size=32)
    return test_loader_poison_MLM, num_normal_removed_POISON, removed_words_poison, sent_removed_word_dict_poison

def get_processed_clean_data(all_clean_MLM, clean_data, bar):
    
    processed_data_MLM = []
    num_normal_removed_CLEAN = []
    ALL_removed_words = []
    ALL_sent_removed_word_dict = []

    data = [item[0] for item in clean_data]
        
    for i, SCORE_li in tqdm(enumerate(all_clean_MLM)):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]

        #########################################################################################

        # Average of the SCORES
        SCORE_without_whole_sentence = SCORE_li[:-1]
        avg_of_SCORE = sum(SCORE_without_whole_sentence) / len(SCORE_without_whole_sentence)

        # Suspicion Score
        processed_SCORE_li_neg = [score - avg_of_SCORE for score in SCORE_li][:-1]
        processed_SCORE_li = [abs(score) for score in processed_SCORE_li_neg]
        
        #########################################################################################

        flag_li = []
        for score in processed_SCORE_li:
            if bar <= score:            # 0: one removing
                flag_li.append(0)
            else:                       # 1: one keeping
                flag_li.append(1)

        sent, filtered_sent_length, removed_clean_words, sent_removed_word_dict_clean = get_processed_sent(flag_li, orig_split_sent)

        # Number of clean words removed
        num_clean_removed = len(orig_split_sent) - filtered_sent_length

        processed_data_MLM.append((sent, clean_data[i][1]))

        ####################################################################################
        ALL_removed_words.append(removed_clean_words)
        ALL_sent_removed_word_dict.append(sent_removed_word_dict_clean)
        num_normal_removed_CLEAN.append(num_clean_removed)
        ####################################################################################

    test_clean_loader_MLM = packDataset_util.get_loader(processed_data_MLM, shuffle=False, batch_size=32)
    
    return test_clean_loader_MLM, num_normal_removed_CLEAN, ALL_removed_words, ALL_sent_removed_word_dict

##########################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--clean_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--record_file', default='record.log')
    args = parser.parse_args()
    
    # Result log file path
    record_file = args.record_file

    # LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    data_selected = args.data

    # Poisoned Victim Model
    model = torch.load(args.model_path)
    if torch.cuda.is_available():
        model.cuda()

    packDataset_util = packDataset_util_bert()
    
    orig_poison_data = get_orig_poison_data()
    clean_data = read_data(args.clean_data_path)
    clean_raw_sentences = [item[0] for item in clean_data]
    
    if args.data == 'ag':
        list_size = 300
        orig_poison_data = orig_poison_data[:list_size]
        clean_raw_sentences = clean_raw_sentences[:list_size]

    # MLM
    if torch.cuda.is_available():
        print("mx gpu")
        ctxs = [mx.gpu()]
    else:
        print("mx cpu")
        ctxs = [mx.cpu()]
    
    # Get Pytorch BERT MLM Scoring
    mlm_bert_model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-uncased')
    scorer = MLMScorerPT(mlm_bert_model, vocab, tokenizer, ctxs)

    all_MLM = get_SCORES(orig_poison_data, model=mlm_bert_model, scorer=scorer)
    all_clean_MLM = get_SCORES(clean_raw_sentences, model=mlm_bert_model, scorer=scorer)

    ##################################### MLM ######################################

    # Original Method (Having Threshold)
    # Compare with MLM Scoring
    # file_path = args.record_file
    
    file_path = record_file
    f = open(file_path, 'w')
    
    for bar in range(0, 100):
        print("")
        print("bar: ", bar)

        # MLM loaders
        test_loader_poison_loader_MLM, num_normal_removed_POISON, ALL_removed_words_from_poison, ALL_sent_removed_word_dict_poison = prepare_poison_data(all_MLM, orig_poison_data, bar)
        processed_clean_loader_MLM, num_normal_removed_CLEAN, ALL_removed_words_from_clean, ALL_sent_removed_word_dict_clean = get_processed_clean_data(all_clean_MLM, clean_data, bar)

        # Evaluation (accack success rate + clean accuracy)
        success_rate_mlm = evaluaion(test_loader_poison_loader_MLM)
        clean_acc_mlm = evaluaion(processed_clean_loader_MLM)

        ################################################## Print on Screen ###########################################################################

#         print('bar: ', bar)
#         print('attack success rate (MLM): ', success_rate_mlm)
#         print('clean acc (MLM): ', clean_acc_mlm)
#         print("Number of normal words removed (POISON): ", num_normal_removed_POISON)
#         print("Number of normal words removed (CLEAN): ", num_normal_removed_CLEAN)

#         print("")
#         print("*"*89)

#         # Poison Data (only print 100th data)
#         for i, word_dict in enumerate(ALL_sent_removed_word_dict_poison):
#             if ((i+1) % 2 == 0):
#                 for sent, removed_words in word_dict.items():
#                     print("Original Sentence (Poison): ", sent)
#                     print("Removed Words (Poison): ", removed_words)
#         print("")
#         print("*"*89)
#         print("")
#         # Clean Data (only print 100th data)
#         for i, word_dict in enumerate(ALL_sent_removed_word_dict_clean):
#             if ((i+1) % 2 == 0):
#                 for sent, removed_words in word_dict.items():
#                     print("Original Sentence (Clean): ", sent)
#                     print("Removed Words (Clean): ", removed_words)

#         print("*"*89)
#         print("")

        ###################################################### Print to log file (file=f) #######################################################################

        print('bar: ', bar, file=f)
        print('attack success rate (MLM): ', success_rate_mlm, file=f)
        print('clean acc (MLM): ', clean_acc_mlm, file=f)
        print("Number of normal words removed (POISON): ", num_normal_removed_POISON, file=f)
        print("Number of normal words removed (CLEAN): ", num_normal_removed_CLEAN, file=f)

        print("", file=f)
        print("*"*89, file=f)
        print("", file=f)

        # Poison Data (only save 100th data)
        for i, word_dict in enumerate(ALL_sent_removed_word_dict_poison):
            if ((i+1) % 20 == 0):
                for sent, removed_words in word_dict.items():
                    print("Original Sentence (Poison): ", sent, file=f)
                    print("Removed Words (Poison): ", removed_words, file=f)
        
        print("", file=f)
        print("*"*89, file=f)
        print("", file=f)
        
        # Clean Data (only save 100th data)
        for i, word_dict in enumerate(ALL_sent_removed_word_dict_clean):
            if ((i+1) % 20 == 0):
                for sent, removed_words in word_dict.items():
                    print("Original Sentence (Clean): ", sent, file=f)
                    print("Removed Words (Clean): ", removed_words, file=f)

        print('*' * 89, file=f)
        print("", file=f)

        #############################################################################################################################

    f.close()
