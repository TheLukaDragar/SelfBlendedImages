from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils


def facecrop(org_path,save_path,face_detector,face_predictor,period=1,num_frames=10):

    

    cap_org = cv2.VideoCapture(org_path)
    
    croppedfaces=[]
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    
    frame_idxs = np.linspace(0, frame_count_org - 1, frame_count_org//period, endpoint=True, dtype=np.int64)
    for cnt_frame in range(frame_count_org): 
        ret_org, frame_org = cap_org.read()
        height,width=frame_org.shape[:-1]
        if not ret_org:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(org_path)))
            break
        
        if cnt_frame not in frame_idxs:
            continue
        
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)


        faces = face_detector(frame, 1)
        if len(faces)==0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(org_path)))
            continue
        face_s_max=-1
        landmarks=[]
        size_list=[]
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0,y0=landmark[:,0].min(),landmark[:,1].min()
            x1,y1=landmark[:,0].max(),landmark[:,1].max()
            face_s=(x1-x0)*(y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]

        save_path_=os.path.join(save_path,'frames/')
        os.makedirs(save_path_,exist_ok=True)
        image_path=save_path_+str(cnt_frame).zfill(3)+'.png'
        land_path=save_path_+str(cnt_frame).zfill(3)

        land_path=land_path.replace('/frames','/landmarks')

        os.makedirs(os.path.dirname(land_path),exist_ok=True)
        np.save(land_path, landmarks)

        if not os.path.isfile(image_path):
            cv2.imwrite(image_path,frame_org)

    cap_org.release()
    return



if __name__=='__main__':

    dataset = json.load(open('/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/src/vra_metadata.json'))
    og_root='/ceph/hpc/data/st2207-pgp-users/ldragar/original_dataset'
    save_path='/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/data/VRA/'

	#make save if not exist
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print(len(dataset['clips']))
    

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = '/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/src/preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)




    for clip in tqdm(list(dataset['clips'].keys())):
        print(dataset['clips'][clip])
        path_tovid = os.path.join(og_root, clip)
        #add mp4
        path_tovid = path_tovid + '.mp4'

        out_path = os.path.join(save_path, clip)
        print("out_path", out_path)

        #make dir if not exist
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        #check if it exists
        if not os.path.isfile(path_tovid):
            print('not found', path_tovid)
            exit(0)
            
        
        facecrop(path_tovid,save_path=out_path,period=2,face_detector=face_detector,face_predictor=face_predictor)
    
    # movies_path=dataset_path+'videos/'

    # movies_path_list=sorted(glob(movies_path+'*.mp4'))
    # print("{} : videos are exist in {}".format(len(movies_path_list),args.dataset))


    # n_sample=len(movies_path_list)

    # for i in tqdm(range(n_sample)):
    #     folder_path=movies_path_list[i].replace('videos/','frames/').replace('.mp4','/')
       
    #     facecrop(movies_path_list[i],save_path=dataset_path,num_frames=args.num_frames,face_predictor=face_predictor,face_detector=face_detector)
    

    
