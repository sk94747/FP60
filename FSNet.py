from DenseNet.MyDensenetModel import *
from MyYolov5Model import *

class FSNet():
    def __init__(self):
        # one stage detect model
        self.yolov5_model = MyYolov5Model()
        # two stage classify model
        self.densenet_model = None
        # result save dir
        self.save_dir_name = str(int(time.time()))

    def get_dataset_by_family_id(self, family_id):
        root_data_path = "./datasets/FP60_Family/"
        train_data_path = root_data_path + str(family_id) + "/train"
        val_data_path = root_data_path+ str(family_id) + "/val"
        test_data_path = root_data_path + str(family_id) + "/test"
        return train_data_path, val_data_path, test_data_path

    def get_model_by_family_id(self, family_id):
        model_name = "DenseNet_FP60_Family_" + str(family_id) + ".pth"
        model_path = "./DenseNet/FP60_Family_model_save/" + model_name
        # model_path = "./DenseNet/FP60_Model_model_save/" + model_name
        return model_path

    # Extract the pest part of the image through the result of one stage
    def get_pic_from_label(self, img_path, boxes, save_part=False):
        # print("=============== Get Insect Par From Image ===============")
        family_id_list = []
        # Picture name without suffix
        img_name = ntpath.basename(img_path)[:-4]
        insect_part_list = []

        im = cv2.imread(img_path)

        save_path = "./result/" + self.save_dir_name + "/images/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for j in range(0, len(boxes)):
            # [0, 14] → [1, 15]
            family_id = int(boxes[j][5] + 1)
            a = int(boxes[j][0])  # x
            b = int(boxes[j][1])  # y
            c = int(boxes[j][2])  # x_l
            d = int(boxes[j][3])  # y_l

            confidence = boxes[j][4]  # confidence
            cropedIm = im[b:d, a:c]

            # save path of new picture
            img_path = save_path + str(family_id) + "_" + img_name + "_" + str(j) + "_" + str(float(confidence)*1000)[:4] + ".jpg"

            family_id_list.append(family_id)
            insect_part_list.append(cropedIm)

            if save_part:
                cv2.imwrite(img_path, cropedIm)

        return family_id_list, insect_part_list

    def detect_once(self, image_path):
        # print("=============== One Stage Image Detect ===============")
        result = []
        boxes = []
        # first stage: insect detect
        one_stage_result = self.yolov5_model.one_stage_image_detect(image_path, self.save_dir_name)
        if len(one_stage_result) > 0:
            # second stage: insect classify
            family_id_list, img_path_list = self.get_pic_from_label(image_path, one_stage_result)
            # print("=============== Two Stage Image classify  ===============")
            for i in range(len(img_path_list)):
                family_id = family_id_list[i]

                train_data_path, val_data_path, test_data_path = self.get_dataset_by_family_id(family_id)
                # create the densenet model by family_id
                self.densenet_model = MyDensenetModel(train_data_path, val_data_path, test_data_path)
                self.densenet_model.model = self.densenet_model.init_model(self.get_model_by_family_id(family_id))
                # test the image
                img = Image.fromarray(img_path_list[i])
                res = self.densenet_model.test_once(img)
                result.append(res)
                boxes.append(one_stage_result[i][:4])
        return result, boxes
