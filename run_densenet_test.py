from DenseNet.MyDensenetModel import *

def get_dataset_by_family_id(family_id):
    root_data_path = "./datasets/FP60_Family/"
    train_data_path = root_data_path + str(family_id) + "/train"
    val_data_path = root_data_path + str(family_id) + "/val"
    test_data_path = root_data_path + str(family_id) + "/test"
    return train_data_path, val_data_path, test_data_path

if __name__ == '__main__':
    family_id = 1
    train_data_path, val_data_path, test_data_path = get_dataset_by_family_id(family_id)
    model = MyDensenetModel(train_data_path, val_data_path, test_data_path)

    model_path = "./DenseNet/FP60_Family_model_save/DenseNet_FP60_Family_1.pth"

    img_path = "./image/1-1.jpg"
    img = Image.open(img_path)
    model.model = model.init_model(model_path)
    result = model.test_once(img)

    print(result)