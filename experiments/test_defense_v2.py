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

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    # print("")
    # print("*"*100)
    # print("")
    # print("Filtered word: ", split_sent[pos])
    # print("")
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

##################################### ONION scoring ######################################

def get_SCORES(data, model, scorer):
    all_PPL = []
    all_MLM = []

    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        single_sent_MLM_score = []

        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)

            if sent_length < 3:
               single_sent_PPL.append(0)
               single_sent_MLM_score.append(0)
               # print("")
               # print("SKIP THIS SENTENCE")         
               # print(split_sent)      
               # print("")
               continue

            ppl = LM(processed_sent)
            score = scorer.score_sentences([processed_sent])
            score = score[0]
            score = -score

            # print("")
            # print("Original Sentence: ", split_sent)
            # print("FilteredSentence: ", processed_sent)
            # print("")
            # print("PPL: ", ppl)
            # print("MLM Score: ", score)
            # print("")
            # print("*"*100)

            single_sent_PPL.append(LM(processed_sent))
            single_sent_MLM_score.append(score)

        all_PPL.append(single_sent_PPL)
        all_MLM.append(single_sent_MLM_score)

    #assert len(all_PPL) == len(data)
    return all_PPL, all_MLM


def get_processed_sent(flag_li, orig_sent):
    sent = []

    # print("Original Sentence: ", orig_sent)

    for i, word in tqdm(enumerate(orig_sent)):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
        # else:
        #     print("")
        #     print("Final Trigger Removed: ", word)
        #     print("")

    # print("Filtered Sentence: ", sent)
    # print("*"*100)
    # print("")

    # Just to evaluate clean data
    filtered_sent_length = len(sent)
    join_sent = ' '.join(sent)

    return join_sent, filtered_sent_length

def get_processed_poison_data(all_PPL, all_MLM, data, bar):
    processed_data_PPL = []
    processed_data_MLM = []

    score_list = [all_PPL, all_MLM]
    score_index = 0

    for j, all_score in tqdm(enumerate(score_list)):
        if j == 0:
            score_name = "PPL"
            # print("")
            # print("*"*100)
            # print("j: ", j)
            # print("Score name: ", score_name)
        else:
            score_name = "MLM Score"
            # print("")
            # print("*"*100)
            # print("j: ", j)
            # print("Score_name: ", score_name)


        for i, SCORE_li in tqdm(enumerate(all_score)):
            orig_sent = data[i]
            orig_split_sent = orig_sent.split(' ')[:-1]
            
            print(len(SCORE_li))
            # assert len(orig_split_sent) == len(SCORE_li) - 1

            whole_sentence_SCORE = SCORE_li[-1]

            #########################################################################################

            # Average of the SCORES
            SCORE_without_whole_sentence = SCORE_li[:-1]
            avg_of_SCORE = sum(SCORE_without_whole_sentence) / len(SCORE_without_whole_sentence)

            # Suspicion Score
            processed_SCORE_li_neg = [score - avg_of_SCORE for score in SCORE_li][:-1]
            processed_SCORE_li = [abs(score) for score in processed_SCORE_li_neg]

            #########################################################################################

            # print("POISON DATA")
            # print("bar: ", bar)
            # print("")
            # print("Average: ", avg_of_SCORE)
            # print("")
            # print("Calculated {} score: {}".format(score_name, SCORE_without_whole_sentence))
            # print("")
            # print("AVERAGE - SCORE difference list: ", processed_SCORE_li)
            # print("")

            flag_li = []
            for score in processed_SCORE_li:
                if bar <= score:        # 0: one removing
                    flag_li.append(0)
                else:                   # 1: one keeping
                    flag_li.append(1)

            # print("")
            # print("flag: ", flag_li)
            # print("flag length: ", len(flag_li))
            # print("")
            # assert len(flag_li) == len(orig_split_sent)
            sent, filtered_sent_length = get_processed_sent(flag_li, orig_split_sent)

            if j == 0:
                # print("")
                # print("*"*100)
                # print("Append in PPL list (POISON)")
                # print("*"*100)
                # print("")
                processed_data_PPL.append((sent, args.target_label))
            else:
                # print("")
                # print("*"*100)
                # print("Append in MLM list (POISON)")
                # print("*"*100)
                # print("")
                processed_data_MLM.append((sent, args.target_label))

    # assert len(all_PPL) == len(processed_data)
    return processed_data_PPL, processed_data_MLM


def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence

def prepare_poison_data(all_PPL, all_MLM, orig_poison_data, bar):
    test_data_poison_ONION, test_data_poison_MLM = get_processed_poison_data(all_PPL, all_MLM, orig_poison_data, bar=bar)

    test_loader_poison_ONION = packDataset_util.get_loader(test_data_poison_ONION, shuffle=False, batch_size=32)
    test_loader_poison_MLM = packDataset_util.get_loader(test_data_poison_MLM, shuffle=False, batch_size=32)

    return test_loader_poison_ONION, test_loader_poison_MLM

def get_processed_clean_data(all_clean_PPL, all_clean_MLM, clean_data, bar):
    processed_data_PPL= []
    processed_data_MLM = []

    num_clean_removed_PPL = []
    num_clean_removed_MLM = []

    data = [item[0] for item in clean_data]
    score_list = [all_clean_PPL, all_clean_MLM]

    for j, score_type in tqdm(enumerate(score_list)):
        if j == 0:
            score_name = "PPL"
            # print("")
            # print("*"*100)
            # print("j: ", j)
            # print("Score name: ", score_name)
        else:
            score_name = "MLM Score"
            # print("")
            # print("*"*100)
            # print("j: ", j)
            # print("Score_name: ", score_name)
        
        for i, SCORE_li in tqdm(enumerate(score_type)):
            orig_sent = data[i]
            orig_split_sent = orig_sent.split(' ')[:-1]
            # assert len(orig_split_sent) == len(SCORE_li) - 1
            whole_sentence_PPL = SCORE_li[-1]

            # Suspicion Word Score
            processed_SCORE_li = [ppl - whole_sentence_PPL for ppl in SCORE_li][:-1]

            #########################################################################################

            # Average of the SCORES
            SCORE_without_whole_sentence = SCORE_li[:-1]
            avg_of_SCORE = sum(SCORE_without_whole_sentence) / len(SCORE_without_whole_sentence)

            # Suspicion Score
            processed_SCORE_li_neg = [score - avg_of_SCORE for score in SCORE_li][:-1]
            processed_SCORE_li = [abs(score) for score in processed_SCORE_li_neg]
            
            #########################################################################################

            # print("POISON DATA")
            # print("bar: ", bar)
            # print("")
            # print("Average: ", avg_of_SCORE)
            # print("")
            # print("Calculated {} score: {}".format(score_name, SCORE_without_whole_sentence))
            # print("")
            # print("AVERAGE - SCORE difference list: ", processed_SCORE_li)
            # print("")
            
            flag_li = []
            for score in processed_SCORE_li:
                if bar <= score:            # 0: one removing
                    flag_li.append(0)
                else:                       # 1: one keeping
                    flag_li.append(1)

            # print("")
            # print("flag: ", flag_li)
            # print("flag length: ", len(flag_li))
            # print("")

            # assert len(flag_li) == len(orig_split_sent)
            sent, filtered_sent_length = get_processed_sent(flag_li, orig_split_sent)

            # Number of clean words removed
            num_clean_removed = len(orig_split_sent) - filtered_sent_length
            # print("Number of clean words removed: ", num_clean_removed)
            # print("")

            if j == 0:
                # print("")
                # print("*"*100)
                # print("Append in PPL list (CLEAN)")
                # print("*"*100)
                # print("")
                processed_data_PPL.append((sent, clean_data[i][1]))
                num_clean_removed_PPL.append(num_clean_removed)
            else:
                # print("")
                # print("*"*100)
                # print("Append in MLM list (CLEAN)")
                # print("*"*100)
                # print("")
                processed_data_MLM.append((sent, clean_data[i][1]))
                num_clean_removed_MLM.append(num_clean_removed)

    # assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader_PPL = packDataset_util.get_loader(processed_data_PPL, shuffle=False, batch_size=32)
    test_clean_loader_MLM = packDataset_util.get_loader(processed_data_MLM, shuffle=False, batch_size=32)
    
    return test_clean_loader_PPL, test_clean_loader_MLM, num_clean_removed_PPL ,num_clean_removed_MLM

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

    LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    data_selected = args.data

    # Poisoned Victim Model
    model = torch.load(args.model_path)
    if torch.cuda.is_available():
        model.cuda()

    packDataset_util = packDataset_util_bert()
    
    orig_poison_data = get_orig_poison_data()
    clean_data = read_data(args.clean_data_path)
    clean_raw_sentences = [item[0] for item in clean_data]
    
    # list_size = 1
    # orig_poison_data = orig_poison_data[:list_size]
    # clean_raw_sentences = clean_raw_sentences[:list_size]

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

    # PPL
    all_PPL, all_MLM = get_SCORES(orig_poison_data, model=mlm_bert_model, scorer=scorer)
    all_clean_PPL, all_clean_MLM = get_SCORES(clean_raw_sentences, model=mlm_bert_model, scorer=scorer)

    ##################################### ONION vs. MLM ######################################

    # Original Method (Having Threshold)
    # Compare with MLM Scoring
    # file_path = args.record_file
    
    file_path = record_file
    f = open(file_path, 'w')

    for bar in range(15, 21):
        print("")
        print("bar: ", bar)

        # ONION + MLM loaders
        test_loader_poison_loader_ONION, test_loader_poison_loader_MLM = prepare_poison_data(all_PPL, all_MLM, orig_poison_data, bar)
        processed_clean_loader_ONION, processed_clean_loader_MLM, num_clean_removed_PPL, num_clean_removed_MLM = get_processed_clean_data(all_clean_PPL, all_clean_MLM, clean_data, bar)

        # Evaluation (accack success rate + clean accuracy)
        success_rate_onion = evaluaion(test_loader_poison_loader_ONION)
        clean_acc_onion = evaluaion(processed_clean_loader_ONION)
        success_rate_mlm = evaluaion(test_loader_poison_loader_MLM)
        clean_acc_mlm = evaluaion(processed_clean_loader_MLM)

        print('bar: ', bar)
        print('attack success rate (ONION): ', success_rate_onion)
        print('clean acc (ONION): ', clean_acc_onion)
        print("Number of normal words removed (ONION): ", num_clean_removed_PPL)
        print("")
        print('attack success rate (MLM): ', success_rate_mlm)
        print('clean acc (MLM): ', clean_acc_mlm)
        print("Number of normal words removed (MLM): ", num_clean_removed_MLM)


        print('bar: ', bar, file=f)
        print('attack success rate (ONION): ', success_rate_onion, file=f)
        print('clean acc (ONION): ', clean_acc_onion, file=f)
        print("Number of normal words removed (ONION): ", num_clean_removed_PPL, file=f)
        print("", file=f)
        print('attack success rate (MLM): ', success_rate_mlm, file=f)
        print('clean acc (MLM): ', clean_acc_mlm, file=f)
        print("Number of normal words removed (MLM): ", num_clean_removed_MLM, file=f)

        print('*' * 89, file=f)

    f.close()
