import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metric


def ROC(fpr, tpr, tprs, set_name):
    colors = ['lightskyblue', 'springgreen', 'lightpink', 'yellow', 'coral']
    plt.clf()
    aucs = []
    for i in range(len(fpr)):
        fpr_ = fpr[i]
        tpr_ = tpr[i]
        plt.plot(fpr_, tpr_, colors[i], alpha=0.3, label='ROC fold {} (AUC {:.3f})'.format(i, round(metric.auc(fpr_, tpr_), 3)))
        aucs.append(metric.auc(fpr_, tpr_))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metric.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(round(mean_auc, 3), round(std_auc, 3)),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label='$\pm$ 1 std. dev.')
    plt.plot([0, 0], [0, 0])
    plt.legend(loc='lower right')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curve ({})'.format(set_name))
    plt.savefig('ROC_{}.png'.format(set_name))


v_acc, v_auc, t_acc, t_auc = [], [], [], []
v_sen, v_spe, t_sen, t_spe = [], [], [], []
v_fpr, v_tpr, t_fpr, t_tpr = [], [], [], []
tprs_val, tprs_test = [], []    
mean_fpr_val = np.linspace(0, 1, 100)
mean_fpr_test = np.linspace(0, 1, 100)
for tf in range(5):
    with open('{}/perform_ws{}_seed{}_hd{}_tf{}.pickle'.format(save_path, ws, seed, hd, tf), 'rb') as f:
        performance = pickle.load(f)
    [val_acc, val_auc, f1_val, test_acc, test_auc, f1_test, conf_tr, conf_val, conf_test, gt_val, gt_test, sftmax_val, sftmax_test] = performance
    # print(val_acc, val_auc, test_acc, test_auc)
    v_acc.append(val_acc)
    v_auc.append(val_auc)
    t_acc.append(test_acc)
    t_auc.append(test_auc)
    sens_val = conf_val[1][1]/(conf_val[1][1]+conf_val[1][0])
    spec_val = conf_val[0][0]/(conf_val[0][0]+conf_val[0][1])
    sens_test = conf_test[1][1]/(conf_test[1][1]+conf_test[1][0])
    spec_test = conf_test[0][0]/(conf_test[0][0]+conf_test[0][1])
    v_sen.append(sens_val)
    v_spe.append(spec_val)
    t_sen.append(sens_test)
    t_spe.append(spec_test)
    conf_tr_sum += conf_tr
    conf_vl_sum += conf_val
    conf_ts_sum += conf_test
    
    fpr_val, tpr_val, _ = metric.roc_curve(gt_val, sftmax_val)
    fpr_test, tpr_test, _ = metric.roc_curve(gt_test, sftmax_test)

    v_fpr.append(fpr_val)
    v_tpr.append(tpr_val)
    t_fpr.append(fpr_test)
    t_tpr.append(tpr_test)
    tprs_val.append(np.interp(mean_fpr_val, fpr_val, tpr_val))
    tprs_test.append(np.interp(mean_fpr_test, fpr_test, tpr_test))

ROC(v_fpr, v_tpr, tprs_val, 'validation')
ROC(t_fpr, t_tpr, tprs_test, 'test')
    

