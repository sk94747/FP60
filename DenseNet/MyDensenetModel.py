import copy
import os
import random
import sys

import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import numpy as np
from PIL import Image
from torchvision import transforms, datasets

from torchvision import models

from utils import myutils, cal_utils
from utils.logger import Logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 图片处理机制
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MyDensenetModel:
    def __init__(self, train_data_path, val_data_path, test_data_path):
        # train & val & test dataset load by ImageFolder
        self.train_data = datasets.ImageFolder(root=train_data_path, transform=data_transform)
        self.val_data = datasets.ImageFolder(root=val_data_path, transform=data_transform)
        self.test_data = datasets.ImageFolder(root=test_data_path, transform=data_transform)
        # the message of dataset from path
        self.dataset_name = train_data_path.split("/")[-3]
        self.dataset_id = train_data_path.split("/")[-2]
        # epoch
        self.EPOCH = 300
        self.current_epoch = 0
        # batch size
        self.BATCH_SIZE = 32
        # learning rate
        self.LR = 1e-4
        # set loader
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.BATCH_SIZE, shuffle=True)

        # get total classes
        self.N_classes = len(self.train_data.classes)
        # check cuda
        self.use_cuda = torch.cuda.is_available()
        # Loss Function
        self.loss_func = nn.CrossEntropyLoss()

        # load model
        self.model = self.init_model()
        self.best_model = self.init_model()

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        # the current accuracy and max accuracy
        self.current_acc = 0
        self.max_acc = 0

    # init model or load model
    def init_model(self, load_model_path=""):
        # use the pretrained model
        model = models.densenet121(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 500)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(500, self.N_classes)),
            ('output', nn.Softmax(dim=1))
        ]))

        model.classifier = classifier

        if not load_model_path == "":
            if self.use_cuda:
                model.load_state_dict(torch.load(load_model_path))
            else:
                model.load_state_dict(torch.load(load_model_path, map_location='cpu'), strict=False)

        if self.use_cuda:
            model = model.cuda()

        return model

    # train the model
    def train_model(self, is_save=False):
        self.model.train()
        print("========== DenseNet Train Begin ==========")
        print("tain dataset num：", len(self.train_data))
        print("val dataset num：", len(self.val_data))
        print("test dataset num：", len(self.test_data))
        print("the batch size：", self.BATCH_SIZE)
        print("total classes：", self.N_classes)
        print("classes message：", self.train_data.classes)
        print("total epoch：", self.EPOCH)
        print('------------------------------------')

        if self.use_cuda:
            self.model = self.model.cuda()

        # train
        for epoch in range(self.EPOCH):
            self.current_epoch += 1
            for step, (b_x, b_y) in enumerate(self.train_loader):
                if self.use_cuda:
                    b_x, b_y = b_x.cuda(), b_y.cuda()
                output = self.model(b_x)
                loss = self.loss_func(output, b_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # validate current model
            if (epoch + 1) % 10 == 0:
                self.val_model()

        # save model
        if is_save:
            print("========= Train Finish and Save Model ==========")
            model_name = "DenseNet_" + self.dataset_name + "_" + self.dataset_id + ".pth"
            save_model_dir = "./DenseNet/" + self.dataset_name + "_model_save"
            save_path = save_model_dir + "/" + model_name
            print('model save path：', save_path)
            torch.save(self.model.state_dict(), save_path)
            #torch.save(self.model.classifier.state_dict(), save_path)
            print("========= Model Save Success ==========")

    # validate model
    def val_model(self):
        self.model.eval()

        n_correct = 0
        for step, (t_x, t_y) in enumerate(self.val_loader):
            if self.use_cuda:
                t_x = t_x.cuda()
            test_output = self.model(t_x)
            if self.use_cuda:
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()
            n_correct += float((pred_y == t_y.data.numpy()).astype(int).sum())

        acc = (n_correct / len(self.test_data))

        if acc >= self.max_acc:
            self.max_acc = acc
            self.best_model = copy.deepcopy(self.model)
            print('the %d epoch val_acc is %.2f%%  and the max_acc is %.2f%%, the best model is update !! ' % (self.current_epoch, (acc * 100), (self.max_acc * 100)))
        else:
            print('the %d epoch val_acc is %.2f%%  and the max_acc is %.2f%%' % (self.current_epoch, (acc * 100), (self.max_acc * 100)))

    # test model
    def test_model(self):
        self.best_model.eval()
        print("========= Model Test Begin ==========")
        print("test dataset num：", len(self.test_data))

        # Create an empty confusion matrix
        conf_matrix = torch.zeros(self.N_classes, self.N_classes)

        time_start = time.time()
        n_correct = 0

        for step, (t_x, t_y) in enumerate(self.test_loader):
            if self.use_cuda:
                t_x = t_x.cuda()

            test_output = self.best_model(t_x)

            if self.use_cuda:
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()

            # update confusion matrix
            conf_matrix = myutils.confusion_matrix(pred_y, labels=t_y.data.numpy(), conf_matrix=conf_matrix)

            n_correct += float((pred_y == t_y.data.numpy()).astype(int).sum())

        time_end = time.time()
        print('Test time cost %.2f s' % (time_end - time_start))
        print("Total true count = ", n_correct)
        acc = (n_correct / len(self.test_data))
        print('The final_acc is %.2f' % (acc * 100), "%")


        # plot the confusion matrix
        plt_save_path = "./conf_matrix.jpg"
        myutils.plot_confusion_matrix(conf_matrix.numpy(), classes=self.train_data.classes, normalize=False, title='DenseNet confusion matrix', save_path=plt_save_path)
        print(conf_matrix)
        # Calculate accuracy, recall rate, F1 score through confusion matrix
        cal_utils.my_cal(self.train_data.class_to_idx, conf_matrix)

    # test model
    def tmp_test_model(self):
        self.best_model.eval()
        print("========= Model Test Begin ==========")
        print("test dataset num：", len(self.test_data))

        # Create an empty confusion matrix
        conf_matrix = torch.zeros(self.N_classes, self.N_classes)

        time_start = time.time()
        n_correct = 0

        for step, (t_x, t_y) in enumerate(self.test_loader):
            if self.use_cuda:
                t_x = t_x.cuda()

            test_output = self.best_model(t_x)

            if self.use_cuda:
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()

            # update confusion matrix
            conf_matrix = myutils.confusion_matrix(pred_y, labels=t_y.data.numpy(), conf_matrix=conf_matrix)

            n_correct += float((pred_y == t_y.data.numpy()).astype(int).sum())

        time_end = time.time()
        print('Test time cost %.2f s' % (time_end - time_start))
        print("Total true count = ", n_correct)
        acc = (n_correct / len(self.test_data))
        print('The final_acc is %.2f' % (acc * 100), "%")

        # plot the confusion matrix
        plt_save_path = "./conf_matrix.jpg"
        myutils.plot_confusion_matrix(conf_matrix.numpy(), classes=self.train_data.classes, normalize=False,
                                      title='DenseNet confusion matrix', save_path=plt_save_path)
        print(conf_matrix)
        # Calculate accuracy, recall rate, F1 score through confusion matrix
        cal_utils.my_cal(self.train_data.class_to_idx, conf_matrix)

    # test one image
    def test_once(self, test_img):
        self.model.eval()

        img = data_transform(test_img)
        img = torch.unsqueeze(img, 0)

        if self.use_cuda:
            img = img.cuda()

        test_output = self.model(img)
        if self.use_cuda:
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        else:
            pred_y = torch.max(test_output, 1)[1].data.numpy()
        classes_dict = dict(zip(self.train_data.class_to_idx.values(), self.train_data.class_to_idx.keys()))

        # print(new_dict)
        return classes_dict[int(pred_y)]

    # used to fix random seed
    def set_seed(self, seed):
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
