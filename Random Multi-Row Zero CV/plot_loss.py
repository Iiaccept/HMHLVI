import matplotlib.pyplot as plt
from datetime import datetime

def plot_loss(epoch_cv,epoch,k_fold):
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = 'black'
    x1= list(range(1,epoch+1))

    for k in range(k_fold):
        y1=epoch_cv[k]
        label1=f'{k+1}_fold_loss'
        plt.plot(x1,y1,label=label1)
    
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend()
    
    current_datetime = datetime.now()
    current_date_str = current_datetime.strftime('%Y_%m_%d')
    print(current_date_str)
    filename1=f'./results/k_fold_loss_{current_date_str}.png' 
    plt.savefig(filename1,dpi=300,bbox_inches= 'tight')
    plt.show()