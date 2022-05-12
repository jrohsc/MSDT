import numpy as np
from matplotlib import pyplot as plt
import random 

# def result_visualization(defense_type, data_type, threshold_list, attack_success_rate_list, clean_acc_list):
def result_visualization():
    threshold_list = list(range(1, 100+1))
    attack_success_rate_list = []
    clean_acc_list = []
    defense_type = 'onion'
    data_type = 'sst'
    for i in range(100): attack_success_rate_list.append(random.randint(10,30))
    for i in range(100): clean_acc_list.append(random.randint(50, 90))

    print(len(threshold_list), len(attack_success_rate_list), len(clean_acc_list))
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(13, 8)

    fontsize = 10
    plt.subplot_tool()
    plt.suptitle('Defense method: "{}" ; Data type: "{}"'.format(defense_type, data_type))
    
    ax[0].plot(threshold_list, attack_success_rate_list, 'r-o')
    ax[0].set_xlim([-20, 120])
    ax[0].set_ylim([0, 100])
    ax[0].set_title('Attack Success Rate per Threshold', fontsize=fontsize)
    ax[0].set_xlabel('Threshold', fontsize=fontsize)
    ax[0].set_ylabel('Attack Success Rate (%)', fontsize=fontsize)

    ax[1].plot(threshold_list, clean_acc_list, 'b-o')
    ax[1].set_xlim([-20, 120])
    ax[1].set_ylim([0, 100])
    ax[1].set_title('Clean Accuracy per Threshold', fontsize=fontsize)
    ax[1].set_xlabel('Threshold', fontsize=fontsize)
    ax[1].set_ylabel('Clean Accuracy (%)', fontsize=fontsize)

    plt.savefig('graphs/{} ({})_graph.png'.format(defense_type, data_type), dpi=300)

    # plt.show()