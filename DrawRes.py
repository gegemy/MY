import numpy as np
import matplotlib.pyplot as plt
import os

# def draw_loss():
    
def get_loss(input_f):
    loss_real = []
    loss_syn = []
    match_dis_loss = []
    pre_epoch = -1
    pre_outer = -1
    pre_dis_loss = 1000000
    pre_real_loss = 1000000
    pre_syn_loss = 1000000
    with open(input_f) as f:
        for line in f.readlines():
            if line.startswith('Epoch') and len(line.strip().split(',')) == 3:
                tmp = line.strip().split(',')[-1]
                loss = float(tmp.split(' ')[-1])
                epoch = int(line.strip().split(',')[0].split(' ')[-1])
                outer = int(line.strip().split(',')[1].split(' ')[-1])
                if epoch > pre_epoch:
                    pre_epoch = epoch
                    pre_outer = -1
                    pre_dis_loss = 1000000
                    pre_real_loss = 1000000
                    pre_syn_loss = 1000000
                if len(tmp.strip().split(' ')) == 2:
                    if epoch == pre_epoch and outer == 0:
                        match_dis_loss.append(loss)
                        pre_dis_loss = loss
                    if epoch == pre_epoch and outer > pre_outer and loss < pre_dis_loss:
                        match_dis_loss.pop()
                        match_dis_loss.append(loss)
                        pre_dis_loss = loss
                else:
                    mdl = str(tmp.split(' ')[-2])
                    if mdl == 'real':
                        if epoch == pre_epoch and outer == 0:
                            loss_real.append(loss)
                            pre_real_loss = loss
                        if epoch == pre_epoch and outer > pre_outer and loss < pre_real_loss:
                            loss_real.pop()
                            loss_real.append(loss)
                            pre_real_loss = loss
                    elif mdl == 'syn':
                        if epoch == pre_epoch and outer == 0:
                            loss_syn.append(loss)
                            pre_syn_loss = loss
                        if epoch == pre_epoch and outer > pre_outer and loss < pre_syn_loss:
                            loss_syn.pop()
                            loss_syn.append(loss)
                            pre_syn_loss = loss
                    else:
                        exit()
    return loss_real, loss_syn, match_dis_loss

def get_acc(input_f):
    TestAcc = []
    ESTestAcc = []
    BestValAcc = []
    with open(input_f) as f:
        for line in f.readlines():
            if line.startswith('---'):
                line = line.strip().split(',')
                TestAcc.append(float(line[0].split(':')[-1].strip()))
                ESTestAcc.append(float(line[1].split(':')[-1].strip()))
                BestValAcc.append(float(line[-1].strip().split(' ')[2].strip()))
    return TestAcc, ESTestAcc, BestValAcc

def get_mean_std(TestAcc, ESTestAcc, BestValAcc):
    testAcc = np.array(TestAcc)
    test_mean_acc = np.mean(testAcc)
    test_std = np.std(testAcc)
    print(test_mean_acc, test_std)
    estestAcc = np.array(ESTestAcc)
    estest_mean_acc = np.mean(estestAcc)
    estest_std = np.std(estestAcc)
    print(estest_mean_acc, estest_std)
    bestValAcc = np.array(BestValAcc)
    bestval_mean_acc = np.mean(bestValAcc)
    bestval_std = np.std(bestValAcc)
    print(bestval_mean_acc, bestval_std)
                        
def draw_loss(loss_real, loss_syn, save_base_path, dataset):
    plt.figure(figsize=(8,5))
    assert len(loss_real) == len(loss_syn)
    x = np.arange(0, len(loss_real))
    plt.plot(x, loss_real, label='Loss Real', color='#B4292C', marker='o', linewidth=2)
    plt.plot(x, loss_syn, label='Loss Syn', color='#C18D04', marker='x', linewidth=2)
    
    plt.ylabel('Loss Value')
    plt.xlabel('# Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_base_path, '{}_loss.png'.format(dataset)))
    
def draw_acc(TestAcc, ESTestAcc, BestValAcc, save_base_path, dataset):
    plt.figure(figsize=(8,5))
    x = np.arange(0, len(TestAcc))
    plt.plot(x, TestAcc, label='TestAcc', color='#B4292C', marker='o', linewidth=3)
    plt.plot(x, ESTestAcc, label='Early-Stopping-TestAcc', color='#C18D04', marker='x', linewidth=3)
    plt.plot(x, BestValAcc, label='Best ValAcc', color='#11346A', marker='*', linewidth=3)
    
    plt.ylabel('Accuracy')
    plt.xlabel('')
    plt.legend()
    plt.savefig(os.path.join(save_base_path, '{}_Acc.png'.format(dataset)))   
    
def draw_match_dis(match_dis_loss, save_base_path, dataset):
    plt.figure(figsize=(8,5))
    x = np.arange(0, len(match_dis_loss))
    plt.plot(x, match_dis_loss, label='Match loss', color='#11346A', marker='*', linewidth=2)
    
    plt.ylabel('Match Distance')
    plt.xlabel('# Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_base_path, '{}_match_dis.png'.format(dataset)))
                            
if __name__=="__main__":
    res_base_path = '/home/Exp/3/MY/MY/res'
    save_base_path = '/home/Exp/3/MY/MY/pic'
    dataset_list = ['arxiv']
    file_list = ['1114_res.txt']
    
    for dataset in dataset_list:
        for f in file_list:
            res_f = os.path.join(res_base_path, dataset, f)
            loss_real, loss_syn, match_dis_loss = get_loss(res_f)
            
            draw_loss(loss_real, loss_syn, save_base_path, dataset)
            draw_match_dis(match_dis_loss, save_base_path, dataset)
            
            TestAcc, ESTestAcc, BestValAcc = get_acc(res_f)
            get_mean_std(TestAcc, ESTestAcc, BestValAcc)
            print(TestAcc, ESTestAcc, BestValAcc)
            draw_acc(TestAcc, ESTestAcc, BestValAcc, save_base_path, dataset)