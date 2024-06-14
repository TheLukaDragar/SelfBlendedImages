from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd


def init_ff(phase,level='frame',n_frames=8):
	dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'
	

	image_list=[]
	label_list=[]

	
	
	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	return image_list,label_list

def init_vra(phase,level='frame',n_frames=8):
	dataset_path='/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/data/VRA/'
	

	image_list=[]
	label_list=[]

	
	
	# folder_list = sorted(glob(dataset_path+'*'))
	# filelist = []
	list_dict = json.load(open(f'/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/src/vra_metadata.json','r'))

	for clip in list_dict['clips'].keys():
		vid_data = list_dict['clips'][clip]
		# print(vid_data)
		if vid_data['type'] == phase or ( phase == 'trainval' and vid_data['type'] in ['train','val'] ) or ( phase == 'test' and vid_data['type'] in ['test1','test2','test3'] ):
			path_to_frames = os.path.join(dataset_path, clip, 'frames')
			#list all frames in the folder
			frames = sorted(glob(path_to_frames+'/*.png'))

			#check if no frames
			if len(frames) == 0:
				print("no frames")
				print(clip)
				continue


			# #linspace for n_frames
			# findexes = np.linspace(0,len(frames)-1,n_frames, dtype=np.int32)
			# #select frames


			

			# #add to image list 
			# for i in findexes:
			# 	#check if index err
				
			# 	frame = frames[i]
				
			# 	#add full path
			# 	image_list.append(frame)
			# 	label_list.append(float(vid_data['label']))

			#append only middle frame

			
			# if not os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')):
			# 	print('missing: ', image_list[i].replace('/frames/',path_lm).replace('.png','.npy'))
			# if not os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy')):
			# 	print('missing: ', image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))

			#check if  moddle_frame it has face landmarks and retina

			i = 0
			while True:	
				middle_frame = frames[len(frames)//2+i]

				path_lm='/landmarks/' 
				path_retina='/retina/'
				if not os.path.isfile(middle_frame.replace('/frames/',path_lm).replace('.png','.npy')):
					print('missing: ', middle_frame.replace('/frames/',path_lm).replace('.png','.npy'))
					i+=1
					continue

				if not os.path.isfile(middle_frame.replace('/frames/','/retina/').replace('.png','.npy')):
					print('missing: ', middle_frame.replace('/frames/','/retina/').replace('.png','.npy'))
					i+=1
					continue

				break


			


			image_list.append(middle_frame)

			label_list.append(float(vid_data['label']))

			



	

	# for i in list_dict:
	# 	filelist+=i
	# folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	# if level =='video':
	# 	return "error"
	# 	label_list=[0]*len(folder_list)
	# 	return folder_list,label_list
	# for i in range(len(folder_list)):
	# 	# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
	# 	images_temp=sorted(glob(folder_list[i]+'/*.png'))
	# 	if n_frames<len(images_temp):
	# 		images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
	# 	image_list+=images_temp
	# 	label_list+=[0]*len(images_temp)

	return image_list,label_list


def init_ff_real(phase,level='frame',n_frames=8):
	dataset_path='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/FaceForensics++/original_videos/original_sequences/youtube/raw/version1-ARNES/'
	

	image_list=[]
	label_list=[]

	#list all dirs in frames dir
	frame_path = os.path.join(dataset_path, 'frames')
	# folder_list = sorted(glob(frame_path+'/*'))

	# # print(folder_list)

	# #take random 100 for val

	# val_list = np.random.choice(folder_list, 100, replace=False)
	# train_list = [i for i in folder_list if i not in val_list]

	# print(len(val_list))
	# print(len(train_list))

	# #to list 
	# val_list = list(val_list)
	
	# #save to ff_real_metadata.json # mark as val or train
	# ff_real_metadata = {}
	# ff_real_metadata['val'] = val_list
	# ff_real_metadata['train'] = train_list

	# with open('ff_real_metadata.json', 'w') as outfile:
	# 	json.dump(ff_real_metadata, outfile)


	# #load from ff_real_metadata.json
	list_dict = json.load(open(f'/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/ff_real_metadata.json','r'))

	# #get list of files
	if phase == 'trainval':
		filelist = list_dict['train'] + list_dict['val']
	else:

		filelist = list_dict[phase]

	print(len(filelist))

	# #loop through files
	for i in filelist:
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(i+'/*.png'))
		# print(i)
		if n_frames<=len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
			#check if these have landmarks and retina
			imgs_ok = []
			for img in images_temp:
				path_lm='/landmarks/' 
				path_retina='/retina/'
				# print("checking: ", img)
				if not os.path.isfile(img.replace('/frames/',path_lm).replace('.png','.npy')):
					print('missing: ', img.replace('/frames/',path_lm).replace('.png','.npy'))
					continue

				if not os.path.isfile(img.replace('/frames/','/retina/').replace('.png','.npy')):
					print('missing: ', img.replace('/frames/','/retina/').replace('.png','.npy'))
					continue
				imgs_ok.append(img)

			if len(imgs_ok) == n_frames:
				image_list+=imgs_ok
				label_list+=[0]*len(imgs_ok)
			else:
				print("found missing landmarks or retina padding with last frame")
				#padd with the same frame
				imgs_ok = imgs_ok + [imgs_ok[-1]]*(n_frames-len(imgs_ok))
				image_list+=imgs_ok
				label_list+=[0]*len(imgs_ok)


	return image_list,label_list


if __name__ == '__main__':
	# init_ff('train')
	# x,y =init_vra('train')
	# print(x[:10])
	# print("len",len(x))
	x,y =init_ff_real('train')
	print(x[:10])
	print("len",len(x))
	