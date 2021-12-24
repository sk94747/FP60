from DenseNet.MyDensenetModel import *

def get_dataset_by_family_id(family_id):
    root_data_path = "./DenseNet/FP60/"
    train_data_path = root_data_path + str(family_id) + "/train"
    val_data_path = root_data_path+ str(family_id) + "/val"
    test_data_path = root_data_path + str(family_id) + "/test"
    return train_data_path, val_data_path, test_data_path

if __name__ == '__main__':
    family_id = 1
    train_data_path, val_data_path, test_data_path = get_dataset_by_family_id(family_id)
    model = MyDensenetModel(train_data_path, val_data_path, test_data_path)
    # train the model and save model
    model.train_model()
    # test the model
    model.test_model()