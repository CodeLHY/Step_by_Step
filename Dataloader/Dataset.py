from torch.utils.data import Dataset
import numpy as np
import cv2
import os
def get_ids(path_txt,mode):
    txt = open(os.path.join(path_txt,mode)+".txt").read()
    IDs = [int(id) for id in txt.split(",")]
    return IDs
def get_data(IDs, cams, path_cams,size):
    """
        获取数据
    :param IDs:  所需要的数据的ID
    :param cams:  从那些cam中获取数据
    :param path_cams:  cams的上一级目录，其内容为： --cam1 --cam2.。。
    :param size:  读取的数据要缩放为多大
    :return:  已经集结好了的图片数据(NCHW)、label、所属的cam、
    """
    data = []
    label = []
    cam_data = []
    modal = []
    for cam in cams:
        path_data = os.path.join(path_cams, cam)
        for id in os.listdir(path_data):
            if int(id) in IDs:
                for img in os.listdir(os.path.join(path_data,id)):
                    image = cv2.imread(os.path.join(path_data,id,img))
                    # cv2.imshow("origin", image)
                    image = cv2.resize(image, size)
                    # cv2.imshow("resized",image)
                    # cv2.waitKey(10000)
                    # cv2.destroyAllWindows()
                    data.append(image)
                    label.append(int(id))
                    cam_data.append(int(cam[-1]))
                    if int(cam[-1]) in [3,6]:
                        modal.append(0)
                    else:
                        modal.append(1)

    return np.array(data),np.array(label),np.array(modal),np.array(cam_data)
def save_data(data,label,modal,cams,path,suffix):
    """

    :param data:  要保存的数据
    :param label:  要保存的label
    :param cams:  要保存的cams
    :param path:  要保存的文件夹路径
    :param suffix:  要保存的名称
    :return:
    """
    np.save(os.path.join(path, "processed_new", "data_" + suffix + ".npy"), data)
    np.save(os.path.join(path, "processed_new", "labels_" + suffix + ".npy"), label)
    np.save(os.path.join(path, "processed_new", "cams_" + suffix + ".npy"), cams)
    np.save(os.path.join(path, "processed_new", "modal_" + suffix + ".npy"), modal)
def pre_process_data(path_cams,path_config,size):
    cam_RGBs = ["cam1","cam2","cam4","cam5"]
    cam_IRs = ["cam3","cam6"]
    IDs_train = get_ids(path_config,"train_id")
    IDs_test = get_ids(path_config, "test_id")
    IDs_val = get_ids(path_config, "val_id")
    data_train_RGB, label_train_RGB, modal_train_RGB, cam_train_RGB = get_data(IDs_train, cam_RGBs,path_cams, size)
    data_train_IR, label_train_IR, modal_train_IR, cam_train_IR = get_data(IDs_train, cam_IRs, path_cams,size)
    data_test_RGB, label_test_RGB, modal_test_RGB, cam_test_RGB = get_data(IDs_test, cam_RGBs, path_cams,size)
    data_test_IR, label_test_IR, modal_test_IR, cam_test_IR = get_data(IDs_test, cam_IRs, path_cams,size)
    data_val_RGB, label_val_RGB, modal_val_RGB, cam_val_RGB = get_data(IDs_val, cam_RGBs, path_cams,size)
    data_cal_IR, label_val_IR, modal_val_IR, cam_val_IR = get_data(IDs_val, cam_IRs, path_cams,size)
    save_data(data_train_RGB, label_train_RGB, modal_train_RGB, cam_train_RGB,path_cams,"train_RGB")
    save_data(data_train_IR, label_train_IR, modal_train_IR, cam_train_IR, path_cams, "train_IR")
    save_data(data_test_RGB, label_test_RGB, modal_test_RGB, cam_test_RGB, path_cams, "test_RGB")
    save_data(data_test_IR, label_test_IR, modal_test_IR, cam_test_IR, path_cams, "test_IR")
    save_data(data_val_RGB, label_val_RGB, modal_val_RGB, cam_val_RGB, path_cams, "val_RGB")
    save_data(data_cal_IR, label_val_IR, modal_val_IR, cam_val_IR, path_cams, "val_IR")
def get_dict(path_data):
    """
    仅针对sysu数据集
    :param path_data: 上层数据集所在的位置， 下层为--cam1 --cam2 。。。
    :return: 找到所有的ID，并且将他们重新排序形成一个新的ID便于处理，为了后续的转换，又制作一个dict，可以将旧的ID转换为对应的新的ID
    """
    id_all_data = []
    cams = ["cam1","cam2","cam3","cam4","cam5","cam6"]
    for cam in cams:
        for ids in os.listdir(os.path.join(path_data,cam)):
            id_all_data.append(int(ids))
    ids_old = sorted(set(id_all_data))
    old2new_id = {old:new for (new,old) in enumerate(ids_old)}
    return old2new_id
class dataset_sbs(Dataset):
    def __init__(self, path_data, mode, dict_label, transform):
        self.dict = dict_label
        self.transform = transform
        self.mode = mode
        if mode == "train":
            self.data_RGB = np.load(path_data+"/data_train_RGB.npy")
            self.label_RGB = np.load(path_data+"/labels_train_RGB.npy")
            self.cam_RGB = np.load(path_data+"/cams_train_RGB.npy")
            self.modal_RGB = np.load(path_data+"/modal_train_RGB.npy")
            self.data_IR = np.load(path_data + "/data_train_IR.npy")
            self.label_IR = np.load(path_data + "/labels_train_IR.npy")
            self.cam_IR = np.load(path_data + "/cams_train_IR.npy")
            self.modal_IR = np.load(path_data + "/modal_train_IR.npy")
            self.data_all = np.concatenate((self.data_RGB,self.data_IR),0)
            self.label_all_old = np.concatenate((self.label_RGB,self.label_IR),0)
            self.label_all = np.array([self.dict[str(label_old)] for label_old in self.label_all_old.tolist()])
            self.cam_all = np.concatenate((self.cam_RGB,self.cam_IR),0)
            self.modal_all = np.concatenate((self.modal_RGB,self.modal_IR),0)
        elif mode == "query":
            self.data_all = np.load(path_data + "/data_val_RGB.npy")
            self.label_all = np.load(path_data + "/labels_val_RGB.npy")
            self.cam_all = np.load(path_data + "/cams_val_RGB.npy")
            self.modal_all = np.load(path_data + "/modal_val_RGB.npy")
        elif mode == "gall":
            self.data_all = np.load(path_data + "/data_val_IR.npy")
            self.label_all = np.load(path_data + "/labels_val_IR.npy")
            self.cam_all = np.load(path_data + "/cams_val_IR.npy")
            self.modal_all = np.load(path_data + "/modal_val_IR.npy")
        elif mode == "test":
            self.data_RGB = np.load(path_data + "/data_test_RGB.npy")
            self.label_RGB = np.load(path_data + "/labels_test_RGB.npy")
            self.cam_RGB = np.load(path_data + "/cams_test_RGB.npy")
            self.modal_RGB = np.load(path_data + "/modal_test_RGB.npy")
            self.data_IR = np.load(path_data + "/data_test_IR.npy")
            self.label_IR = np.load(path_data + "/labels_test_IR.npy")
            self.cam_IR = np.load(path_data + "/cams_test_IR.npy")
            self.modal_IR = np.load(path_data + "/modal_test_IR.npy")
            self.data_all = np.concatenate((self.data_RGB, self.data_IR), 0)
            self.label_all_old = np.concatenate((self.label_RGB, self.label_IR), 0)
            self.label_all = np.array([self.dict[str(label_old)] for label_old in self.label_all_old.tolist()])
            self.cam_all = np.concatenate((self.cam_RGB, self.cam_IR), 0)
            self.modal_all = np.concatenate((self.modal_RGB, self.modal_IR), 0)
    def __getitem__(self, index):
        data = self.data_all[index, :, :, :]
        label = self.label_all[index]
        cam = self.cam_all[index]
        modal = self.modal_all[index]
        # print(data.shape)
        return self.transform(data), label, cam, modal
    def __len__(self):
        return self.label_all.shape[0]

# debug
# IDs = get_ids("K:\SYSU-MM01\multiple modality\SYSU-MM01\SYSU-MM01\exp", "train_id")
# cams = ["cam3","cam6"]
# # data,label,modal,cam_data = get_data(IDs,cams ,"K:\SYSU-MM01\multiple modality\SYSU-MM01\SYSU-MM01",(80,256))
# pre_process_data("K:\SYSU-MM01\multiple modality\SYSU-MM01\SYSU-MM01","K:\SYSU-MM01\multiple modality\SYSU-MM01\SYSU-MM01\exp",(80,256))
# dict_using = get_dict("K:\SYSU-MM01\multiple modality\SYSU-MM01\SYSU-MM01")
# dataset = dataset_sbs("K:\SYSU-MM01\multiple modality\SYSU-MM01\SYSU-MM01\processed_new","train",dict_using)
# print("ok")