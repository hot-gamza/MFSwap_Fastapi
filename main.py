from fastapi import FastAPI, File, UploadFile
# image classification
from fastapi.responses import FileResponse
import shutil
from torchvision import datasets, models, transforms
import torch
import cv2
import numpy as np

import paddle
import argparse
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel

from gfpgan import GFPGANer
from PIL import Image

def get_id_emb_from_image(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

def image_test_multi_face(source_path, target_path, set_device, source_aligned_images, target_aligned_images):
    #paddle.set_device("gpu" if args.use_gpu else 'cpu')
    paddle.set_device("cpu" if torch.cuda.is_available() else 'cpu')
    faceswap_model = FaceSwap(set_device)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    #target_path = args.target_img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

    start_idx = target_path.rfind('/')
    if start_idx > 0:
        target_name = target_path[target_path.rfind('/'):]
    else:
        target_name = target_path
    origin_att_img = cv2.imread(target_path) # align 작업을 거치지 않은 원본 target image
    #id_emb, id_feature = get_id_emb(id_net, base_path + '_aligned.png')

    # print('source_aligned_images : ',source_aligned_images)
    # print('target_aligned_images : ',target_aligned_images)

    for idx, target_aligned_image in enumerate(target_aligned_images):
        id_emb, id_feature = get_id_emb_from_image(id_net, source_aligned_images[idx % len(source_aligned_images)][0])
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()
        #print(target_aligned_image.shape)

        att_img = cv2paddle(target_aligned_image[0])
        #import time
        #start = time.perf_counter()

        res, mask = faceswap_model(att_img)
        #print('process time :{}', time.perf_counter() - start)
        res = paddle2cv(res)

        #dest[landmarks[idx][0]:landmarks[idx][1],:] =

        back_matrix = target_aligned_images[idx % len(target_aligned_images)][1]
        mask = np.transpose(mask[0].numpy(), (1, 2, 0))
        origin_att_img = dealign(res, origin_att_img, back_matrix, mask)

    cv2.imwrite(os.path.join('./swap_img', os.path.basename(target_name.format(idx))), origin_att_img)
    result_img_path = os.path.join('./swap_img', os.path.basename(target_name.format(idx)))
    gfpgan_gogo(result_img_path)

def faces_align(landmarkModel, image_path, image_size=224):
    aligned_imgs =[]
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    
    # print('img_list : ', img_list)
    for path in img_list:
        # print('img_path : ', path)
        img = cv2.imread(path)
        landmarks = landmarkModel.gets(img)
        # print('landmark : ', landmarks)
        for landmark in landmarks:
            if landmark is not None:
                aligned_img, back_matrix = align_img(img, landmark, image_size)
                aligned_imgs.append([aligned_img, back_matrix])
    return aligned_imgs

def gfpgan_gogo(img):
    img = Image.open(img)
    original_img = img.copy()
    np_img = np.array(img)

    device = 'cpu'
    model = GFPGANer(model_path='./models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=device)
    np_img_bgr = np_img[:, :, ::-1]
    _, _, gfpgan_output_bgr = model.enhance(np_img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_img = gfpgan_output_bgr[:, :, ::-1]

    restored_img = Image.fromarray(np_img)
    # restored_img.show()
    result_img = Image.blend(
        original_img, restored_img, 1
    )

    result_img.show()
    base_path = './results/'
    result_img_np = np.array(result_img)
    result_img_rgb = result_img_np[:, :, ::-1]
    cv2.imwrite(base_path + 'gfpgan_img.png', result_img_rgb)

##################################################################################################################
app = FastAPI()

set_device = "cpu" if torch.cuda.is_available() else 'cpu'

@app.get('/')
def root():
    return {"message":"ok"}

@app.post('/mf_swap')
def sendimg(source_file:UploadFile=File(...), target_file:UploadFile=File(...)):
    # 파일 불러오기 및 저장
    print(source_file.filename, target_file.filename)
    source_path = f'files/source/{source_file.filename}'
    target_path = f'files/target/{target_file.filename}'

    with open(source_path, 'w+b') as buffer:
        shutil.copyfileobj(source_file.file, buffer)
    print('source : ', source_file.filename)

    with open(target_path, 'w+b') as buffer:
        shutil.copyfileobj(target_file.file, buffer)
    print('target : ', target_file.filename)    

    # landmark 검출 및 face swap 처리
    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    source_aligned_images = faces_align(landmarkModel, source_path)
    target_aligned_images = faces_align(landmarkModel, target_path, 224)
    
    image_test_multi_face(source_path, target_path, set_device, source_aligned_images, target_aligned_images)
    # # image = cv2.imread(f'files/{file.filename}')
        
    return {'result':'ok'}