
from DenseNet.MyDensenetModel import *
from MyYolov5Model import *

class BHIC():
    def __init__(self):
        # one stage detect model
        self.yolov5_model = MyYolov5Model()
        # two stage classify model
        self.densenet_model = None
        # result save dir
        self.save_dir_name = str(int(time.time()))

    def get_dataset_by_family_id(self, family_id):
        root_data_path = "./DenseNet/FP60/"
        train_data_path = root_data_path + str(family_id) + "/train"
        val_data_path = root_data_path+ str(family_id) + "/val"
        test_data_path = root_data_path + str(family_id) + "/test"
        return train_data_path, val_data_path, test_data_path

    def get_model_by_family_id(self, family_id):
        model_name = "DenseNet_FP60_" + str(family_id) + ".pth"
        model_path = "./DenseNet/FP60_save_model/" + str(family_id) + "/" + model_name
        return model_path

    # Extract the pest part of the image through the result of one stage
    def get_pic_from_label(self, img_path, label_path):
        print("=============== Get Insect Par From Image ===============")
        family_id_list = []
        img_path_list = []
        # Picture name without suffix
        img_name = ntpath.basename(img_path)[:-4]

        dir = open(label_path)
        lines = dir.readlines()
        lists = []
        for line in lines:
            lists.append(line.split())

        picture = cv2.imread(img_path)
        im = cv2.imread(img_path)

        save_path = "./result/" + self.save_dir_name + "/images/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for j in range(0, len(lists)):
            # [0, 14] → [1, 15]
            family_id = int(lists[j][0]) + 1
            a = lists[j][1]  # x
            b = lists[j][2]  # y
            c = lists[j][3]  # x_l
            d = lists[j][4]  # y_l
            confidence = lists[j][5] # confidence

            # Restore the normalized coordinates in the label to the size of the picture
            e = int((float(a) * picture.shape[1]) - (float(c) * picture.shape[1] / 2))
            f = int((float(b) * picture.shape[0]) - (float(d) * picture.shape[0] / 2))
            q = int((float(a) * picture.shape[1]) + (float(c) * picture.shape[1] / 2))
            s = int((float(b) * picture.shape[0]) + (float(d) * picture.shape[0] / 2))
            cropedIm = im[f:s, e:q]
            # save path of new picture
            img_path = save_path + str(family_id) + "_" + img_name + "_" + str(j) + "_" + str(float(confidence)*1000)[:4] + ".jpg"

            family_id_list.append(family_id)
            img_path_list.append(img_path)

            print("save_img:", img_path)
            cv2.imwrite(img_path, cropedIm)

        return family_id_list, img_path_list

    def detect_once(self, image_path):
        print("=============== One Stage Image Detect ===============")
        result = []
        # 一阶段目标检测
        res_dic = self.yolov5_model.one_stage_image_detect(image_path, self.save_dir_name)

        if len(res_dic['insect_res']) > 0:
            print("insect_dic:", res_dic['insect_res'])
            # 二阶段提取图片
            family_id_list, img_path_list = self.get_pic_from_label(res_dic['res_img_path'], res_dic['res_label_path'])
            print("=============== Two Stage Image classify  ===============")
            for i in range(len(img_path_list)):
                family_id = family_id_list[i]
                print("family_id:", family_id)
                train_data_path, val_data_path, test_data_path = self.get_dataset_by_family_id(family_id)
                # create the densenet model by family_id
                self.densenet_model = MyDensenetModel(train_data_path, val_data_path, test_data_path)
                self.densenet_model.model = self.densenet_model.init_model(self.get_model_by_family_id(family_id))
                # test the image
                result.append(self.densenet_model.test_once(img_path_list[i]))
        return result
