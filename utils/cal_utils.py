#####模块说明######
'''
根据传入的文件true_label和predict_label来求模型预测的精度、召回率和F1值，另外给出微观和宏观取值。
powered by wangjianing 2019.3.2
'''
import numpy as np



def getLabelData(file_dir):
    '''
    模型的预测生成相应的label文件，以及真实类标文件，根据文件读取并加载所有label
    1、参数说明：
        file_dir：加载的文件地址。
        文件内数据格式：每行包含两列，第一列为编号1,2，...，第二列为预测或实际的类标签名称。两列以空格为分隔符。
        需要生成两个文件，一个是预测，一个是实际类标，必须保证一一对应，个数一致
    2、返回值：
        返回文件中每一行的label列表，例如['true','false','false',...,'true']
    '''
    labels = []
    with open(file_dir, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            labels.append(i.strip().split(' ')[1])
    return labels


def getLabel2idx(labels):
    '''
    获取所有类标
    返回值：label2idx字典，key表示类名称，value表示编号0,1,2...
    '''
    label2idx = dict()
    for i in labels:
        if i not in label2idx:
            label2idx[i] = len(label2idx)
    return label2idx

#
# def buildConfusionMatrix(predict_file, true_file):
#     '''
#     针对实际类标和预测类标，生成对应的矩阵。
#     矩阵横坐标表示实际的类标，纵坐标表示预测的类标
#     矩阵的元素(m1,m2)表示类标m1被预测为m2的个数。
#     所有元素的数字的和即为测试集样本数，对角线元素和为被预测正确的个数，其余则为预测错误。
#     返回值：返回这个矩阵numpy
#     '''
#     true_labels = getLabelData(true_file)
#     predict_labels = getLabelData(predict_file)
#     label2idx = getLabel2idx(true_labels)
#     confMatrix = np.zeros([len(label2idx), len(label2idx)], dtype=np.int32)
#     for i in range(len(true_labels)):
#         true_labels_idx = label2idx[true_labels[i]]
#         predict_labels_idx = label2idx[predict_labels[i]]
#         confMatrix[true_labels_idx][predict_labels_idx] += 1
#     return confMatrix, label2idx


def calculate_all_precision(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    precision = round(100 * float(correct_sum) / float(total_sum), 2)
    return precision


def calculate_label_precision(confMatrix, labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    precision = 0
    if label_total_sum != 0:
        precision = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return precision


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(precision, recall):
    if (precision + recall) == 0:
        return 0
    return round(2 * precision * recall / (precision + recall), 2)


def my_cal(label2idx, conf_matrix):
    '''
    该为主函数，可将该函数导入自己项目模块中
    打印准确率、精确率、召回率、F1值的格式可自行设计
    '''
    # 读取文件并转化为混淆矩阵,并返回label2idx
    confMatrix, label2idx = conf_matrix, label2idx
    print("============= result calculate =====================")
    # 计算总个数
    total_sum = confMatrix.sum().data.numpy()
    # 全局的准确率
    all_precision = calculate_all_precision(confMatrix)
    # 每个列表的准确率（列表）
    label_precision = []
    # 每个类别的召回率（列表）
    label_recall = []

    # 输出图片总数与总类别数
    print('total_sum =', total_sum, ',label_num =', len(label2idx), '\n')

    for i in label2idx:
        # print(i, end=' ')
        label_precision.append(calculate_label_precision(confMatrix, label2idx[i]))
        label_recall.append(calculate_label_recall(confMatrix, label2idx[i]))
        for j in label2idx:
            labelidx_i = label2idx[i]
            label2idx_j = label2idx[j]
        #     print('  ', confMatrix[labelidx_i][label2idx_j], end=' ')
        # print('\n')

    print('individual result:\n')
    for ei, i in enumerate(label2idx):
        print(ei, '\t', i, '\t', 'precision =', label_precision[ei], '%,\trecall =', label_recall[ei], '%,\tf1 =',
              calculate_f1(label_precision[ei], label_recall[ei]), '%')
    p = round(np.array(label_precision).sum() / len(label_precision), 2)
    r = round(np.array(label_recall).sum() / len(label_precision), 2)
    print('\nMACRO-averaged:\nprecision =', p, '%, recall =', r, '%, f1=', calculate_f1(p, r), '%')
    print('final accuracy = ', all_precision, '%')

    return p, r, calculate_f1(p, r), all_precision

