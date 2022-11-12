# -*- coding: utf-8 -*-
from glob import glob
from densefuse_net import DenseFuseNet
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from channel_fusion import channel_f as channel_fusion
from utils import mkdir,Strategy
_tensor = transforms.ToTensor()
_pil_gray = transforms.ToPILImage()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'
model = DenseFuseNet().to(device)
# checkpoint = torch.load('./train_result/H_model_weight_new.pkl')
checkpoint = torch.load('/home/wps/桌面/ImageFusion_Dualbranch_Fusion-master/train_result/H_best.pkl')
# checkpoint = torch.load('./train_result/model_weight_new.pkl')
model.load_state_dict(checkpoint['weight'])


mkdir("result")
test_ir = './1/'
test_vi = './2/'


def load_img(img_path, img_type=''):
    img = Image.open(img_path)
    if img_type == 'gray':
        img = img.convert('L')
    return _tensor(img).unsqueeze(0).to(device)

fusename = ['add']

def test(model):
    i=1
    img_list_ir = os.listdir(test_ir)
    img_num = len(img_list_ir)
    print("Test images num", img_num)
    for img in img_list_ir:
        img=os.path.splitext(img)
        img1_path = test_ir+img[0]+'.jpg'
        img2_path = test_vi+img[0]+'.jpg'

        img1, img2, = load_img(img1_path), load_img(img2_path)
        s_time = time.time()
        feature1, feature2 = model.encoder(img1,isTest=True),model.encoder(img2,isTest=True)
        for name in fusename:
            if name =='channel':
                features = channel_fusion(feature1, feature2,is_test=True)
                out = model.decoder(features).squeeze(0).detach().cpu()
            else:
                with torch.no_grad():
                    fusion_layer = Strategy(name, 1).to(device)
                    feature_fusion = fusion_layer(feature1, feature2)
                    out = model.decoder(feature_fusion).squeeze(0).detach().cpu()
            e_time = time.time() - s_time
            save_name = 'result/'+name+'/fusion'+str(i)+'.jpg'
            mkdir('result/'+name)
            img_fusion = _pil_gray(out)
            img_fusion.save(save_name)
            print("pic:[%d] %.4fs %s"%(i,e_time,save_name))
        i+=1
with torch.no_grad():
    test(model)
