from Dataloader.Dataset import dataset_sbs, get_dict
from model.evaluate import eval_sysu
from Dataloader.save_load_dict import load_dict
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model.Model_sbs import model_sys
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import argparse
from tqdm import tqdm
import os


#  =================================参数的选择====================================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids',default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--batch_size',default=32, type=int, help='size of batch')
parser.add_argument('--num_epoch',default=100, type=int, help='number of epoch')
parser.add_argument('--save_dir',default='/home/hongyang/CODE/Step_by_Step/result/',type=str, help='path to save trained network')
parser.add_argument('--lr_origin', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr_step', default=20, type=int, help='step to change learning rate')
parser.add_argument('--declaration', default="you know what?", type=str, help="the details that you want to write down")
opt = parser.parse_args()
lr_origin = opt.lr_origin
lr_step = opt.lr_step
num_epoch = opt.num_epoch
batch_size = opt.batch_size
name = opt.name
save_dir = opt.save_dir
path_save_net = os.path.join(save_dir, name, "trained_net")
declaration = opt.declaration
#  -------------------------------setting GPU-----------------------------------
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
#  =============================================================================


#  ===========================save the configuration============================
#  将本次训练的配置信息保存在专门用来保存训练结果的文件夹下的 以该次训练的名称命名的文件夹下
os.mkdir(os.path.join(save_dir,name))
file_config = open(os.path.join(save_dir,name,"config.txt"), "w")
config = "gpu_ids:"+str(gpu_ids)+"\n" + "name:" + name +"\n" + "batch_size:" + str(batch_size) +\
         "\n" + "num+epoch:" + str(num_epoch) + "\n" + "lr_origin:" +str(lr_origin) + "\n" + "lr_step:" + str(lr_step) +\
         "\n" + declaration
file_config.write(config)
file_config.close()
#  =============================================================================


#  ==================================loading====================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((300, 80)),
    transforms.Pad(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(300, 80)),
        transforms.ToTensor(),
        normalize
])
# get training data
dict_using = load_dict("/home/hongyang/CODE/Step_by_Step/Data/dict")
dataset = dataset_sbs("/home/hongyang/CODE/Step_by_Step/Data/processed_new/", "train", dict_using, transform_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# get validate data
dataset_query = dataset_sbs("/home/hongyang/CODE/Step_by_Step/Data/processed_new/", "query", dict_using, transform_val)
dataloader_query = DataLoader(dataset_query, batch_size=32, shuffle=False)
dataset_gall = dataset_sbs("/home/hongyang/CODE/Step_by_Step/Data/processed_new/", "gall", dict_using, transform_val)
dataloader_gall = DataLoader(dataset_gall, batch_size=32, shuffle=False)
#  =============================================================================


#  =============================model AND optimizer=============================
model = model_sys()
model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=lr_origin, weight_decay=5e-4, momentum=0.9)
schedule = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
#  =============================================================================


#  =================================losses======================================
loss_id = nn.CrossEntropyLoss()
loss_id = loss_id.cuda()
#  =============================================================================


#  =============================fire  fire  fire！！=============================
#  todo: 看看下面的get val data是否有bug  OK
#  ----------------------------------training-----------------------------------
print("training:")
for epoch in range(num_epoch):
    print("\n--epoch:", epoch)
    for data_batch in tqdm(dataloader,ncols=60):
        model.train()
        # ===========数据拆开，并且放入GPU==========
        data, label, cam, modal = data_batch
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        cam = Variable(cam.cuda())
        modal = Variable(modal.cuda())
        # =======================================
        features, preds = model(data, modal)
        loss = loss_id(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    schedule.step()
#  -----------------------------------------------------------------------------
    #  todo: 看看这个cmc是否正确，有无bug  OK
#  ---------------------------------evaluating----------------------------------
    # when 2 val
    if epoch % 2 == 0:
        print("--evaluating...:")
        with torch.no_grad():
            model.eval()
            gall_feature_all = torch.zeros((32, 512))
            query_feature_all = torch.zeros((32, 512))
            for i, (data_batch_gall, data_batch_query) in enumerate(tqdm(zip(dataloader_gall, dataloader_query),ncols=60)):
                data_gall, label_gall, cam_gall, modal_gall = data_batch_gall
                data_query, label_query, cam_query, modal_query = data_batch_query
                data_gall = Variable(data_gall.cuda())
                data_query = Variable(data_query.cuda())
                cam_gall = Variable(cam_gall.cuda())
                label_gall = Variable(label_gall.cuda())
                modal_gall = Variable(modal_gall.cuda())
                cam_query = Variable(cam_query.cuda())
                label_query = Variable(label_query.cuda())
                modal_query = Variable(modal_query.cuda())
                feature_gall, _ = model(data_gall, modal_gall)
                feature_query, _ = model(data_query, modal_query)
                if i == 0:
                    gall_feature_all = feature_gall
                    gall_label_all = label_gall  # .unsqueeze(0)
                    gall_cam_all = cam_gall  # .unsqueeze(0)
                    gall_modal_all = modal_gall  # .unsqueeze(0)
                    query_feature_all = feature_query
                    query_label_all = label_query  # .unsqueeze(0)
                    query_cam_all = cam_query  # .unsqueeze(0)
                    query_modal_all = modal_query  # .unsqueeze(0)
                else:
                    gall_feature_all = torch.cat((gall_feature_all, feature_gall), dim=0)
                    gall_label_all = torch.cat((gall_label_all, label_gall), dim=0)
                    gall_cam_all = torch.cat((gall_cam_all, cam_gall), dim=0)
                    gall_modal_all = torch.cat((gall_modal_all, modal_gall), dim=0)
                    query_feature_all = torch.cat((query_feature_all, feature_query), dim=0)
                    query_label_all = torch.cat((query_label_all, label_query), dim=0)
                    query_cam_all = torch.cat((query_cam_all, cam_query), dim=0)
                    query_modal_all = torch.cat((query_modal_all, modal_query), dim=0)
            distmat = np.matmul(query_feature_all.cpu(), np.transpose(gall_feature_all.cpu()))
            cmc, mAP = eval_sysu(-distmat, query_label_all.cpu().numpy(), gall_label_all.cpu().numpy(),
                                 query_cam_all.cpu().numpy(), gall_cam_all.cpu().numpy())
            all_cmc = cmc
            all_mAP = mAP
            # print('evaluate at {} epoch'.format(epoch))
            print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19]))
            print('mAP: {:.2%}'.format(mAP))
    # when 10 save the net
    if epoch % 10 == 0:
        dir_save = path_save_net + "epoch" + str(epoch) + ".pth"
        torch.save(model.state_dict(), dir_save)
#  -----------------------------------------------------------------------------
#  =============================================================================
