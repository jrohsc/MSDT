import pandas as pd
import numpy as np

def generate_table(defense_type, data_selected, threshold_list, attack_success_rate_list, clean_acc_list):
    attack_percentage_list = [i * 100 for i in attack_success_rate_list]
    clean_percentage_list = [i * 100 for i in clean_acc_list]

    if data_selected == 'sst-2': orig_clean_acc = 90.88
    elif data_selected == 'offenseval': orig_clean_acc = 81.96
    elif data_selected == 'ag': orig_clean_acc = 93.97
    else: orig_clean_acc = 100
        
    ASR = [100] * len(attack_success_rate_list)
    CACC = [orig_clean_acc] * len(clean_acc_list) 

    delta_ASR = list(np.subtract(np.array(ASR), np.array(attack_percentage_list)))
    delta_CACC = list(np.subtract(np.array(CACC), np.array(clean_percentage_list)))


    df = pd.DataFrame()
    df['threshold'] = pd.Series(threshold_list)
    df['ASR'] = pd.Series(ASR)
    df['delta_ASR'] = pd.Series(delta_ASR)
    df['CACC'] = pd.Series(CACC)
    df['delta_CACC'] = pd.Series(delta_CACC)

    df.to_csv('tables/{} ({})_table.csv'.format(defense_type, data_selected))
