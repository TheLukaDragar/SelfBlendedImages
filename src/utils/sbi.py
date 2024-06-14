# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

from albumentations.pytorch.transforms import ToTensorV2
import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb

import warnings
warnings.filterwarnings('ignore')


import logging

if os.path.isfile('/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/src/utils/library/bi_online_generation.py'):
	sys.path.append('/ceph/hpc/data/st2207-pgp-users/ldragar/SelfBlendedImages/src/utils/library/')
	print('exist library')
	exist_bi=True
else:
	exist_bi=False

# class SelfBlendingParams:
# 			def __init__(self, strength=1.0):
# 				self.strength = strength

# 				mask_x_y_blur_ratio = self.adjust_for_strength(0.25, 1.0, increasing=False)
# 				# print("mask_x_y_blur_ratio: ", mask_x_y_blur_ratio)

# 				self.sb_params = {
# 					'hull_type': 0,  # Can be 0,1,2,3 (dfl_full, extended, components, facehull)
# 					'blend_ratio': self.adjust_for_strength(0.25, 1.0),
# 					'mask_size_h': int(257 * mask_x_y_blur_ratio),
# 					'mask_size_w': 257 ,
# 					# Decrease kernel sizes with higher strength
# 					'kernel_1_size': self.ensure_odd(self.adjust_for_strength_int(3, 17, increasing=False)),
# 					'kernel_2_size': self.ensure_odd(self.adjust_for_strength_int(3, 17, increasing=False)),
# 					# Increase mask offset with higher strength
# 					'x_offset': self.adjust_for_strength_random_flip_sign(0, 0.03, increasing=True),
# 					'y_offset': self.adjust_for_strength_random_flip_sign(0, 0.015, increasing=True),
# 					'scale': self.adjust_for_strength(0.95,1, increasing=False),
# 					'elastic_alpha': self.adjust_for_strength_int(20, 100),
# 					'elastic_sigma': self.adjust_for_strength_int(4, 7),
# 					'prob_landmark_reduction': 0.0,
# 					'prob_transform_src': 0.5,  # AUGMENTATION OTHER THAN SELF BLENDING
# 				}

# 			def ensure_odd(self, value):
# 				"""
# 				Ensures that a value is odd.
				
# 				:param value: The value to ensure is odd.
# 				:return: The value, adjusted to be odd.
# 				"""
# 				if value % 2 == 0:
# 					return value + 1
# 				else:
# 					return value
				
# 			def adjust_for_strength(self, min_value, max_value, increasing=True):
# 				"""
# 				Adjusts a parameter value based on the blending strength.
				
# 				:param min_value: The minimum value of the parameter.
# 				:param max_value: The maximum value of the parameter.
# 				:param increasing: A boolean indicating if the value should increase with strength.
# 				:return: A value adjusted according to the blending strength.
# 				"""
# 				if increasing:
# 					return min_value + (max_value - min_value) * self.strength
# 				else:
# 					return max_value - (max_value - min_value) * self.strength
				

# 			#function that chooses lower or higher value based on random choice
# 			def random_choose_weighted(self, min_value, max_value, increasing=True):
# 				"""
# 				Chooses a value in the range 0.25 to 1 with a distribution more skewed towards higher values,
# 				increasing variance as strength increases.

# 				:return: A value in the range 0.25 to 1 adjusted according to the blending strength.
# 				"""
# 				values = np.linspace(min_value, max_value, 100)  # 100 points between 0.25 and 1
# 				# Increase the exponential weight to enhance skewness
# 				weights = ((values - min_value) / 0.75) ** (3 * self.strength + 1)  # More aggressive exponent
# 				weights /= weights.sum()  # Normalize weights to sum to 1
# 				return np.random.choice(values, p=weights)


# 			def adjust_for_strength_random_flip_sign(self, min_value, max_value, increasing=True):
# 				"""
# 				Adjusts a parameter value based on the blending strength.
				
# 				:param min_value: The minimum value of the parameter.
# 				:param max_value: The maximum value of the parameter.
# 				:param increasing: A boolean indicating if the value should increase with strength.
# 				:return: A value adjusted according to the blending strength.
# 				"""
# 				if increasing:
# 					return min_value + (max_value - min_value) * self.strength * np.random.choice([-1,1])
# 				else:
# 					return max_value - (max_value - min_value) * self.strength * np.random.choice([-1,1])

				
# 			def adjust_for_strength_int(self, min_value, max_value, increasing=True):
# 				"""
# 				Adjusts a parameter value based on the blending strength.
				
# 				:param min_value: The minimum value of the parameter.
# 				:param max_value: The maximum value of the parameter.
# 				:param increasing: A boolean indicating if the value should increase with strength.
# 				:return: A value adjusted according to the blending strength.
# 				"""
# 				if increasing:
# 					return int(min_value + (max_value - min_value) * self.strength)
# 				else:
# 					return int(max_value - (max_value - min_value) * self.strength)



class SelfBlendingParams:
    def __init__(self, strength=1.0):
        self.strength = strength
        self.random_offset = 0.15

        mask_x_y_blur_ratio = self.adjust_for_strength(0.25, 1.0, increasing=False)
        # print("mask_x_y_blur_ratio: ", mask_x_y_blur_ratio)

        self.sb_params = {
            "hull_type": 0,  # Can be 0,1,2,3 (dfl_full, extended, components, facehull)
            "blend_ratio": self.adjust_for_strength_random(0.25, 1.0),
            "mask_size_h": int(257 * mask_x_y_blur_ratio),
            "mask_size_w": 257,
            # Decrease kernel sizes with higher strength
            "kernel_1_size": self.ensure_odd(
                self.adjust_for_strength_int_random(3, 17, increasing=False)
            ),
            "kernel_2_size": self.ensure_odd(
                self.adjust_for_strength_int_random(3, 17, increasing=False)
            ),
            # Increase mask offset with higher strength
            "x_offset": self.adjust_for_strength_random_flip_sign(
                0, 0.03, increasing=True
            ),
            "y_offset": self.adjust_for_strength_random_flip_sign(
                0, 0.015, increasing=True
            ),
            "scale": self.adjust_for_strength_random(0.95, 1, increasing=False),
            "elastic_alpha": self.adjust_for_strength_int(20, 100),
            "elastic_sigma": self.adjust_for_strength_int_random(4, 7),
            "prob_landmark_reduction": 0.0,
            "prob_transform_src": 0.5,  # AUGMENTATION OTHER THAN SELF BLENDING
        }

    def ensure_odd(self, value):
        """
        Ensures that a value is odd.

        :param value: The value to ensure is odd.
        :return: The value, adjusted to be odd.
        """
        if value % 2 == 0:
            return value + 1
        else:
            return value

    def adjust_for_strength(self, min_value, max_value, increasing=True):
        """
        Adjusts a parameter value based on the blending strength.

        :param min_value: The minimum value of the parameter.
        :param max_value: The maximum value of the parameter.
        :param increasing: A boolean indicating if the value should increase with strength.
        :return: A value adjusted according to the blending strength.
        """
        if increasing:
            return min_value + (max_value - min_value) * self.strength
        else:
            return max_value - (max_value - min_value) * self.strength

    def adjust_for_strength_random(self, min_value, max_value, increasing=True):
        """
        Adjusts a parameter value based on the blending strength and introduces randomness.

        :param min_value: The minimum value of the parameter.
        :param max_value: The maximum value of the parameter.
        :param increasing: A boolean indicating if the value should increase with strength.
        :return: A value adjusted according to the blending strength with added randomness.
        """
        if increasing:
            value = min_value + (max_value - min_value) * self.strength
        else:
            value = max_value - (max_value - min_value) * self.strength

        # Add randomness within a specified percentage of the current range
        random_range_percentage =  self.random_offset  # For example, 10%
        random_range = (max_value - min_value) * random_range_percentage
        value = value + np.random.uniform(-random_range, random_range)
        return np.clip(value, min_value, max_value)



    # function that chooses lower or higher value based on random choice
    def random_choose_weighted(self, min_value, max_value, increasing=True):
        """
        Chooses a value in the range 0.25 to 1 with a distribution more skewed towards higher values,
        increasing variance as strength increases.

        :return: A value in the range 0.25 to 1 adjusted according to the blending strength.
        """
        values = np.linspace(min_value, max_value, 100)  # 100 points between 0.25 and 1
        # Increase the exponential weight to enhance skewness
        weights = ((values - min_value) / 0.75) ** (
            3 * self.strength + 1
        )  # More aggressive exponent
        weights /= weights.sum()  # Normalize weights to sum to 1
        return np.random.choice(values, p=weights)

    def adjust_for_strength_random_flip_sign(
        self, min_value, max_value, increasing=True
    ):
        """
        Adjusts a parameter value based on the blending strength.

        :param min_value: The minimum value of the parameter.
        :param max_value: The maximum value of the parameter.
        :param increasing: A boolean indicating if the value should increase with strength.
        :return: A value adjusted according to the blending strength.
        """
        if increasing:
            return min_value + (
                max_value - min_value
            ) * self.strength * np.random.choice([-1, 1])
        else:
            return max_value - (
                max_value - min_value
            ) * self.strength * np.random.choice([-1, 1])

    def adjust_for_strength_int(self, min_value, max_value, increasing=True):
        """
        Adjusts a parameter value based on the blending strength.

        :param min_value: The minimum value of the parameter.
        :param max_value: The maximum value of the parameter.
        :param increasing: A boolean indicating if the value should increase with strength.
        :return: A value adjusted according to the blending strength.
        """
        if increasing:
            return int(min_value + (max_value - min_value) * self.strength)
        else:
            return int(max_value - (max_value - min_value) * self.strength)

    def adjust_for_strength_int_random(self, min_value, max_value, increasing=True):
        """
        Adjusts a parameter value based on the blending strength.

        :param min_value: The minimum value of the parameter.
        :param max_value: The maximum value of the parameter.
        :param increasing: A boolean indicating if the value should increase with strength.
        :return: A value adjusted according to the blending strength.
        """

        val = 0
        if increasing:
            val = int(min_value + (max_value - min_value) * self.strength)
        else:
            val = int(max_value - (max_value - min_value) * self.strength)

        # Add randomness within a specified percentage of the current range
        random_range_percentage = self.random_offset  # For example, 10%
        random_range = (max_value - min_value) * random_range_percentage
        val = val + np.random.uniform(-random_range, random_range)
        val = np.clip(val, min_value, max_value)
        return int(val)

class SBI_Dataset_VRA3_real(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8,my_transform=None):
		
		assert phase in ['train','val','trainval']
		
		image_list,label_list=init_ff_real(phase,'frame',n_frames=n_frames)

		#only first 64
		# image_list=image_list[:64]
		# label_list=label_list[:64]
		
		path_lm='/landmarks/' 
		# path_lm='/retina/'
		prev_len=len(image_list)
		prev_len_label=len(label_list)
		#check missing
		for i in range(len(image_list)):
			if not os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')):
				print('missing: ', image_list[i].replace('/frames/',path_lm).replace('.png','.npy'))
			if not os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy')):
				print('missing: ', image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))
		
		label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]

		print(f'REAL({phase}): {prev_len}->{len(image_list)}')
		print(f'REAL({phase}): {prev_len_label}->{len(label_list)}')

		


		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list
		self.label_list=label_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		self.my_transform=my_transform

		self.sb_params={
			'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
			'blend_ratio': np.random.choice([0.25,0.5,0.75,1,1,1]),  #[0.25,0.5,0.75,1,1,1]
			'mask_size_h':np.random.randint(192,257), #(192,257)
			'mask_size_w':np.random.randint(192,257), #(192,257)
			'kernel_1_size':random.randrange(5,26,2),
			'kernel_2_size': random.randrange(5,26,2),
			'x_offset':np.random.uniform(-0.03,0.03), #from (-0.03,0.03)
			'y_offset':np.random.uniform(-0.015,0.015), #from (-0.015,0.015)
			'scale':np.random.uniform(0.95,1/0.95), #from [0.95,1/0.95]
			'elastic_alpha':50, #50
			'elastic_sigma':7, #7
			'prob_landmark_reduction':0.25, #0.25
			'prob_transform_src':0.5, #0.5 #NOT USED NO AUGMENTATION OTHER THAN SELF BLENDING
		}

		# self.sb_params0={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 0.5,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':224, #(192,257)
		# 	'mask_size_w':224, #(192,257)
		# 	'kernel_1_size':11,
		# 	'kernel_2_size': 11,
		# 	'x_offset':-0.03, #from (-0.03,0.03)
		# 	'y_offset':-0.015, #from (-0.015,0.015)
		# 	'scale':0.95, #from [0.95,1/0.95]
		# 	'elastic_alpha':50, #50
		# 	'elastic_sigma':7, #7
		# 	'prob_landmark_reduction':0.0, #0.25
		# 	'prob_transform_src':0.0, #0.5
		# }

		
		# # Example usage:
		# # strength = 1.0  # Adjust this value between 0.0 and 1.0 to control the blending strength
		# self.sb_params0 = SelfBlendingParams(strength=strength).sb_params
		# print(self.sb_params0)



		# self.sb_params0={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 1.0,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':257, #(192,257) #ko blura masko na kok jo downscala - potem jo itak nazaj upscala na og
		# 	'mask_size_w':257, #(192,257) #pac ce jo downsamplas bo blur drugace deloval na x in y kordinatah
		# 	'kernel_1_size':17,
		# 	'kernel_2_size': 17,
		# 	'x_offset':0.01, #from (-0.03,0.03)
		# 	'y_offset':0, #from (-0.015,0.015)
		# 	'scale':1.0, #from [0.95,1/0.95]
		# 	'elastic_alpha':50, #50   20-100
		# 	'elastic_sigma':7, #7  #kotrolira koliko so maske valovite okoli robov
		# 	'prob_landmark_reduction':0.0, #0.25
		# 	'prob_transform_src':0.0, #0.5
		# }
		# self.sb_params1={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 0.5,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':224+16,
		# 	'mask_size_w':224+16,
		# 	'kernel_1_size':11,
		# 	'kernel_2_size': 11,
		# 	'x_offset':-0.03/2, #from (-0.03,0.03)
		# 	'y_offset':-0.015/2, #from (-0.015,0.015)
		# 	'scale':0.96, #from [0.95,1/0.95]
		# 	'elastic_alpha':45,
		# 	'elastic_sigma':6,
		# 	'prob_landmark_reduction':0.0,
		# 	'prob_transform_src':0.0, #0.5
		# }
		

		

		# print(self.sb_params)



	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			# try:
			filename=self.image_list[idx]
			img=np.array(Image.open(filename))
			landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
			# print(landmark.shape)
			bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
			bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
			iou_max=-1
			for i in range(len(bboxes)):
				iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
				if iou_max<iou:
					bbox=bboxes[i]
					iou_max=iou

			landmark=self.reorder_landmark(landmark)

			# do not flip
			# if self.phase=='train':
			# 	if np.random.rand()<0.5:
			# 		img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
			
			img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=False)

			#random strengh from 0-1
			strength = round(np.random.uniform(0.0, 1.0), 2)
			#round to 2 decimals
			sb_params = SelfBlendingParams(strength=strength).sb_params


			strength = 1 - strength

			#make strenght value from 1-5
			strength = strength * 5 + 1
			#cut if >5
			if strength > 5:
				strength = 5


			print("get_item strength: ", strength)


		

	

			# breakpoint()
			# img_r22,img_f22,mask_f22=self.self_blending(img.copy(),landmark.copy())
			img_r,img_f,mask_f=self.controlled_self_blending(img.copy(),landmark.copy(),sb_params)
			# img_r1,img_f1,mask_f1=self.controlled_self_blending(img.copy(),landmark.copy(),self.sb_params1)

			#DO NOT AUGMENT
			# if self.phase=='train':
			# 	transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
			# 	img_f=transformed['image']
			# 	img_r=transformed['image1']
			# 	img_f1=transformed['image']
			# 	img_r1=transformed['image1']

			


			
				
			
			img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=controlled_crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
			
			img_r=img_r[y0_new:y1_new,x0_new:x1_new]


			
			
			# img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_f1=cv2.resize(img_f1,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_r1=cv2.resize(img_r1,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255


			

			# img_f=img_f.transpose((2,0,1))
			# img_r=img_r.transpose((2,0,1))
			# img_f1=img_f1.transpose((2,0,1))
			# img_r1=img_r1.transpose((2,0,1))

			if self.my_transform is not None:
				# frame = self.transform(image=frame)["image"]
                # pertrubated_frame = self.transform(image=pertrubated_frame)["image"]
				# print("img_f.shape: ", img_f.shape)
				img_f=self.my_transform(image=img_f.astype('float32'))['image']
				img_r=self.my_transform(image=img_r.astype('float32'))['image']
				
				#duplicate mask dims to get 3 channels instead of 1
				#img_f.shape:  (557, 446, 3)
				#mask_f.shape:  (557, 452, 1)\
				mask_f = np.dstack([mask_f.squeeze()] * 3)

				#multiply by 255
				mask_f = mask_f * 255
				

				
				

				# print("mask_f.shape: ", mask_f.shape)
				# #mask_f.shape:  (557, 452, 3, 1)

				# # #remove last dim
				# # mask_f=mask_f[:,:,0]
				# # mask_f1=mask_f1[:,:,0]

				# print("mask_f.shape: ", mask_f.shape)
				

				mask_f=self.my_transform(image=mask_f.astype('float32'))['image']







			flag=False
			# except Exception as e:
			# 	print("Error: ", e)
			# 	idx=torch.randint(low=0,high=len(self),size=(1,)).item()

			#print typr
			# print(type(img_r), type(img_f), type(img_f1))
		
		return img_f,img_r,self.label_list[idx],self.image_list[idx],mask_f,strength,sb_params

	
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask
	

	def controlled_affine(self,img,mask,x_offset=0.03,y_offset=0.015,scale=0.95,elastic_alpha=50,elastic_sigma=7):
		f=alb.Affine(
				translate_percent={'x':x_offset,'y':y_offset},
				scale=scale,
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=elastic_alpha,
				sigma=elastic_sigma,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		# print("mask shape: ", mask.shape)
		return img,mask

	def controlled_self_blending(self,img,landmark,params):

		hull_type=params['hull_type'] # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		blend_ratio=params['blend_ratio']  #[0.25,0.5,0.75,1,1,1]
		mask_size_h=params['mask_size_h'] #(192,257)
		mask_size_w=params['mask_size_w'] #(192,257)
		kernel_1_size=params['kernel_1_size'] #(11,11)
		kernel_2_size=params['kernel_2_size'] #(11,11)
		x_offset=params['x_offset'] #from (-0.03,0.03)
		y_offset=params['y_offset'] #from (-0.015,0.015)
		scale=params['scale'] #from [0.95,1/0.95]
		elastic_alpha=params['elastic_alpha'] #50
		elastic_sigma=params['elastic_sigma'] #7

		prob_landmark_reduction=params['prob_landmark_reduction'] #0.25
		prob_transform_src=params['prob_transform_src'] #0.5


		H,W=len(img),len(img[0])
		if np.random.rand()<prob_landmark_reduction:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			# mask=random_get_hull(landmark,img)[:,:,0]
			mask=get_hull_controlled(landmark,img,hull_type=hull_type)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

	
		source = img.copy()
		if np.random.rand()<prob_transform_src:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.controlled_affine(source,mask,x_offset=x_offset,y_offset=y_offset,scale=scale,elastic_alpha=elastic_alpha,elastic_sigma=elastic_sigma)
		
	
		img_blended,mask=B.controlled_blend(source,img,mask,blend_ratio=blend_ratio,mask_size_h=mask_size_h,mask_size_w=mask_size_w, kernel_1_size=kernel_1_size, kernel_2_size=kernel_2_size)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

	

		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

	

		source, mask = self.randaffine(source,mask)

		
	
		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_f1,img_r,labels,names=zip(*batch)
		data={}
		# print(img_f[0][0])
		# print(img_f[0][1])
		#important keep the order!!!!
		#torch tensor is used to transform tuples given by zip to tensors
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float(),torch.tensor(img_f1).float()],0)
		# print(type(img_r), type(img_f), type(img_f1))

		# print(isinstance(img_r, torch.Tensor), isinstance(img_f, torch.Tensor), isinstance(img_f1, torch.Tensor))
		# data['img']=torch.cat([img_r,img_f,img_f1],0)
		# data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		data['label']=torch.tensor(labels)

		print("data['img'].shape: ", data['img'].shape)
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)
#module load FFmpeg/4.4.2-GCCcore-11.3.0
class SBI_Dataset_VRA2(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8,my_transform=None,fixed_sb_params=False):
		
		assert phase in ['train','val','test1','test2','test3','trainval','test']
		
		image_list,label_list=init_vra(phase,'frame',n_frames=n_frames)

		#only first 64
		# image_list=image_list[:64]
		# label_list=label_list[:64]
		
		path_lm='/landmarks/' 
		# path_lm='/retina/'
		prev_len=len(image_list)
		prev_len_label=len(label_list)
		#check missing
		for i in range(len(image_list)):
			if not os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')):
				print('missing: ', image_list[i].replace('/frames/',path_lm).replace('.png','.npy'))
			if not os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy')):
				print('missing: ', image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))
		
		label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]

		print(f'VRA({phase}): {prev_len}->{len(image_list)}')
		print(f'VRA({phase}): {prev_len_label}->{len(label_list)}')

		


		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list
		self.label_list=label_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		self.my_transform=my_transform
		self.fixed_sb_params=fixed_sb_params

		self.sb_params={
			'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
			'blend_ratio': np.random.choice([0.25,0.5,0.75,1,1,1]),  #[0.25,0.5,0.75,1,1,1]
			'mask_size_h':np.random.randint(192,257), #(192,257)
			'mask_size_w':np.random.randint(192,257), #(192,257)
			'kernel_1_size':random.randrange(5,26,2),
			'kernel_2_size': random.randrange(5,26,2),
			'x_offset':np.random.uniform(-0.03,0.03), #from (-0.03,0.03)
			'y_offset':np.random.uniform(-0.015,0.015), #from (-0.015,0.015)
			'scale':np.random.uniform(0.95,1/0.95), #from [0.95,1/0.95]
			'elastic_alpha':50, #50
			'elastic_sigma':7, #7
			'prob_landmark_reduction':0.25, #0.25
			'prob_transform_src':0.5, #0.5 #NOT USED NO AUGMENTATION OTHER THAN SELF BLENDING
		}

		# self.sb_params0={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 0.5,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':224, #(192,257)
		# 	'mask_size_w':224, #(192,257)
		# 	'kernel_1_size':11,
		# 	'kernel_2_size': 11,
		# 	'x_offset':-0.03, #from (-0.03,0.03)
		# 	'y_offset':-0.015, #from (-0.015,0.015)
		# 	'scale':0.95, #from [0.95,1/0.95]
		# 	'elastic_alpha':50, #50
		# 	'elastic_sigma':7, #7
		# 	'prob_landmark_reduction':0.0, #0.25
		# 	'prob_transform_src':0.0, #0.5
		# }

		
		# # Example usage:
		# # strength = 1.0  # Adjust this value between 0.0 and 1.0 to control the blending strength
		# self.sb_params0 = SelfBlendingParams(strength=strength).sb_params
		# print(self.sb_params0)



		# self.sb_params0={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 1.0,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':257, #(192,257) #ko blura masko na kok jo downscala - potem jo itak nazaj upscala na og
		# 	'mask_size_w':257, #(192,257) #pac ce jo downsamplas bo blur drugace deloval na x in y kordinatah
		# 	'kernel_1_size':17,
		# 	'kernel_2_size': 17,
		# 	'x_offset':0.01, #from (-0.03,0.03)
		# 	'y_offset':0, #from (-0.015,0.015)
		# 	'scale':1.0, #from [0.95,1/0.95]
		# 	'elastic_alpha':50, #50   20-100
		# 	'elastic_sigma':7, #7  #kotrolira koliko so maske valovite okoli robov
		# 	'prob_landmark_reduction':0.0, #0.25
		# 	'prob_transform_src':0.0, #0.5
		# }
		# self.sb_params1={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 0.5,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':224+16,
		# 	'mask_size_w':224+16,
		# 	'kernel_1_size':11,
		# 	'kernel_2_size': 11,
		# 	'x_offset':-0.03/2, #from (-0.03,0.03)
		# 	'y_offset':-0.015/2, #from (-0.015,0.015)
		# 	'scale':0.96, #from [0.95,1/0.95]
		# 	'elastic_alpha':45,
		# 	'elastic_sigma':6,
		# 	'prob_landmark_reduction':0.0,
		# 	'prob_transform_src':0.0, #0.5
		# }
		

		

		# print(self.sb_params)



	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			# try:
			filename=self.image_list[idx]
			img=np.array(Image.open(filename))
			landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
			# print(landmark.shape)
			bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
			bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
			iou_max=-1
			for i in range(len(bboxes)):
				iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
				if iou_max<iou:
					bbox=bboxes[i]
					iou_max=iou

			landmark=self.reorder_landmark(landmark)

			# do not flip
			# if self.phase=='train':
			# 	if np.random.rand()<0.5:
			# 		img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
			
			img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=False)

			#random strengh from 0-1
			strength = round(np.random.uniform(0.0, 1.0), 2)

			if self.fixed_sb_params:
				strength = 0.5


			#round to 2 decimals
			sb_params = SelfBlendingParams(strength=strength).sb_params


			strength = 1 - strength

			#make strenght value from 1-5
			strength = strength * 5 + 1
			#cut if >5
			if strength > 5:
				strength = 5


			print("get_item strength: ", strength)


		

	

			# breakpoint()
			# img_r22,img_f22,mask_f22=self.self_blending(img.copy(),landmark.copy())
			img_r,img_f,mask_f=self.controlled_self_blending(img.copy(),landmark.copy(),sb_params)
			# img_r1,img_f1,mask_f1=self.controlled_self_blending(img.copy(),landmark.copy(),self.sb_params1)

			#DO NOT AUGMENT
			# if self.phase=='train':
			# 	transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
			# 	img_f=transformed['image']
			# 	img_r=transformed['image1']
			# 	img_f1=transformed['image']
			# 	img_r1=transformed['image1']

			


			
				
			
			img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=controlled_crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
			
			img_r=img_r[y0_new:y1_new,x0_new:x1_new]


			
			
			# img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_f1=cv2.resize(img_f1,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_r1=cv2.resize(img_r1,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255


			

			# img_f=img_f.transpose((2,0,1))
			# img_r=img_r.transpose((2,0,1))
			# img_f1=img_f1.transpose((2,0,1))
			# img_r1=img_r1.transpose((2,0,1))

			if self.my_transform is not None:
				# frame = self.transform(image=frame)["image"]
                # pertrubated_frame = self.transform(image=pertrubated_frame)["image"]
				# print("img_f.shape: ", img_f.shape)
				img_f=self.my_transform(image=img_f.astype('float32'))['image']
				img_r=self.my_transform(image=img_r.astype('float32'))['image']
				
				#duplicate mask dims to get 3 channels instead of 1
				#img_f.shape:  (557, 446, 3)
				#mask_f.shape:  (557, 452, 1)\
				mask_f = np.dstack([mask_f.squeeze()] * 3)

				#multiply by 255
				mask_f = mask_f * 255
				

				
				

				# print("mask_f.shape: ", mask_f.shape)
				# #mask_f.shape:  (557, 452, 3, 1)

				# # #remove last dim
				# # mask_f=mask_f[:,:,0]
				# # mask_f1=mask_f1[:,:,0]

				# print("mask_f.shape: ", mask_f.shape)
				

				mask_f=self.my_transform(image=mask_f.astype('float32'))['image']







			flag=False
			# except Exception as e:
			# 	print("Error: ", e)
			# 	idx=torch.randint(low=0,high=len(self),size=(1,)).item()

			#print typr
			# print(type(img_r), type(img_f), type(img_f1))
		
		return img_f,img_r,self.label_list[idx],self.image_list[idx],mask_f,strength,sb_params

	
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask
	

	def controlled_affine(self,img,mask,x_offset=0.03,y_offset=0.015,scale=0.95,elastic_alpha=50,elastic_sigma=7):
		f=alb.Affine(
				translate_percent={'x':x_offset,'y':y_offset},
				scale=scale,
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=elastic_alpha,
				sigma=elastic_sigma,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		# print("mask shape: ", mask.shape)
		return img,mask

	def controlled_self_blending(self,img,landmark,params):

		hull_type=params['hull_type'] # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		blend_ratio=params['blend_ratio']  #[0.25,0.5,0.75,1,1,1]
		mask_size_h=params['mask_size_h'] #(192,257)
		mask_size_w=params['mask_size_w'] #(192,257)
		kernel_1_size=params['kernel_1_size'] #(11,11)
		kernel_2_size=params['kernel_2_size'] #(11,11)
		x_offset=params['x_offset'] #from (-0.03,0.03)
		y_offset=params['y_offset'] #from (-0.015,0.015)
		scale=params['scale'] #from [0.95,1/0.95]
		elastic_alpha=params['elastic_alpha'] #50
		elastic_sigma=params['elastic_sigma'] #7

		prob_landmark_reduction=params['prob_landmark_reduction'] #0.25
		prob_transform_src=params['prob_transform_src'] #0.5


		H,W=len(img),len(img[0])
		if np.random.rand()<prob_landmark_reduction:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			# mask=random_get_hull(landmark,img)[:,:,0]
			mask=get_hull_controlled(landmark,img,hull_type=hull_type)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

	
		source = img.copy()
		# if np.random.rand()<prob_transform_src:
		# 	source = self.source_transforms(image=source.astype(np.uint8))['image']
		# else:
		# 	img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.controlled_affine(source,mask,x_offset=x_offset,y_offset=y_offset,scale=scale,elastic_alpha=elastic_alpha,elastic_sigma=elastic_sigma)
		
	
		img_blended,mask=B.controlled_blend(source,img,mask,blend_ratio=blend_ratio,mask_size_h=mask_size_h,mask_size_w=mask_size_w, kernel_1_size=kernel_1_size, kernel_2_size=kernel_2_size)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

	

		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

	

		source, mask = self.randaffine(source,mask)

		
	
		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_f1,img_r,labels,names=zip(*batch)
		data={}
		# print(img_f[0][0])
		# print(img_f[0][1])
		#important keep the order!!!!
		#torch tensor is used to transform tuples given by zip to tensors
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float(),torch.tensor(img_f1).float()],0)
		# print(type(img_r), type(img_f), type(img_f1))

		# print(isinstance(img_r, torch.Tensor), isinstance(img_f, torch.Tensor), isinstance(img_f1, torch.Tensor))
		# data['img']=torch.cat([img_r,img_f,img_f1],0)
		# data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		data['label']=torch.tensor(labels)

		print("data['img'].shape: ", data['img'].shape)
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)


class SBI_Dataset_VRA(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8,my_transform=None,strength=1.0):
		
		assert phase in ['train','val','test1','test2','test3','trainval','test']
		
		image_list,label_list=init_vra(phase,'frame',n_frames=n_frames)

		#only first 64
		# image_list=image_list[:64]
		# label_list=label_list[:64]
		
		path_lm='/landmarks/' 
		# path_lm='/retina/'
		prev_len=len(image_list)
		prev_len_label=len(label_list)
		#check missing
		for i in range(len(image_list)):
			if not os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')):
				print('missing: ', image_list[i].replace('/frames/',path_lm).replace('.png','.npy'))
			if not os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy')):
				print('missing: ', image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))
		
		label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]

		print(f'VRA({phase}): {prev_len}->{len(image_list)}')
		print(f'VRA({phase}): {prev_len_label}->{len(label_list)}')

		


		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list
		self.label_list=label_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		self.my_transform=my_transform

		self.sb_params={
			'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
			'blend_ratio': np.random.choice([0.25,0.5,0.75,1,1,1]),  #[0.25,0.5,0.75,1,1,1]
			'mask_size_h':np.random.randint(192,257), #(192,257)
			'mask_size_w':np.random.randint(192,257), #(192,257)
			'kernel_1_size':random.randrange(5,26,2),
			'kernel_2_size': random.randrange(5,26,2),
			'x_offset':np.random.uniform(-0.03,0.03), #from (-0.03,0.03)
			'y_offset':np.random.uniform(-0.015,0.015), #from (-0.015,0.015)
			'scale':np.random.uniform(0.95,1/0.95), #from [0.95,1/0.95]
			'elastic_alpha':50, #50
			'elastic_sigma':7, #7
			'prob_landmark_reduction':0.25, #0.25
			'prob_transform_src':0.5, #0.5 #NOT USED NO AUGMENTATION OTHER THAN SELF BLENDING
		}

		# self.sb_params0={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 0.5,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':224, #(192,257)
		# 	'mask_size_w':224, #(192,257)
		# 	'kernel_1_size':11,
		# 	'kernel_2_size': 11,
		# 	'x_offset':-0.03, #from (-0.03,0.03)
		# 	'y_offset':-0.015, #from (-0.015,0.015)
		# 	'scale':0.95, #from [0.95,1/0.95]
		# 	'elastic_alpha':50, #50
		# 	'elastic_sigma':7, #7
		# 	'prob_landmark_reduction':0.0, #0.25
		# 	'prob_transform_src':0.0, #0.5
		# }

		class SelfBlendingParams:
			def __init__(self, strength=1.0):
				self.strength = strength

				mask_x_y_blur_ratio = self.adjust_for_strength(0.25, 1.0, increasing=False)
				print("mask_x_y_blur_ratio: ", mask_x_y_blur_ratio)

				self.sb_params = {
					'hull_type': 0,  # Can be 0,1,2,3 (dfl_full, extended, components, facehull)
					'blend_ratio': self.adjust_for_strength(0.25, 1.0),
					'mask_size_h': int(257 * mask_x_y_blur_ratio),
					'mask_size_w': 257 ,
					# Decrease kernel sizes with higher strength
					'kernel_1_size': self.ensure_odd(self.adjust_for_strength_int(3, 17, increasing=False)),
					'kernel_2_size': self.ensure_odd(self.adjust_for_strength_int(3, 17, increasing=False)),
					# Increase mask offset with higher strength
					'x_offset': self.adjust_for_strength(0, 0.03, increasing=True),
					'y_offset': self.adjust_for_strength(0, 0.015, increasing=True),
					'scale': self.adjust_for_strength(0.95,1, increasing=False),
					'elastic_alpha': self.adjust_for_strength_int(20, 100),
					'elastic_sigma': self.adjust_for_strength_int(4, 7),
					'prob_landmark_reduction': 0.0,
					'prob_transform_src': 0.0,
				}

			def ensure_odd(self, value):
				"""
				Ensures that a value is odd.
				
				:param value: The value to ensure is odd.
				:return: The value, adjusted to be odd.
				"""
				if value % 2 == 0:
					return value + 1
				else:
					return value
				
			def adjust_for_strength(self, min_value, max_value, increasing=True):
				"""
				Adjusts a parameter value based on the blending strength.
				
				:param min_value: The minimum value of the parameter.
				:param max_value: The maximum value of the parameter.
				:param increasing: A boolean indicating if the value should increase with strength.
				:return: A value adjusted according to the blending strength.
				"""
				if increasing:
					return min_value + (max_value - min_value) * self.strength
				else:
					return max_value - (max_value - min_value) * self.strength
				
			def adjust_for_strength_int(self, min_value, max_value, increasing=True):
				"""
				Adjusts a parameter value based on the blending strength.
				
				:param min_value: The minimum value of the parameter.
				:param max_value: The maximum value of the parameter.
				:param increasing: A boolean indicating if the value should increase with strength.
				:return: A value adjusted according to the blending strength.
				"""
				if increasing:
					return int(min_value + (max_value - min_value) * self.strength)
				else:
					return int(max_value - (max_value - min_value) * self.strength)

		# Example usage:
		# strength = 1.0  # Adjust this value between 0.0 and 1.0 to control the blending strength
		self.sb_params0 = SelfBlendingParams(strength=strength).sb_params
		print(self.sb_params0)



		# self.sb_params0={
		# 	'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		# 	'blend_ratio': 1.0,  #[0.25,0.5,0.75,1,1,1]
		# 	'mask_size_h':257, #(192,257) #ko blura masko na kok jo downscala - potem jo itak nazaj upscala na og
		# 	'mask_size_w':257, #(192,257) #pac ce jo downsamplas bo blur drugace deloval na x in y kordinatah
		# 	'kernel_1_size':17,
		# 	'kernel_2_size': 17,
		# 	'x_offset':0.01, #from (-0.03,0.03)
		# 	'y_offset':0, #from (-0.015,0.015)
		# 	'scale':1.0, #from [0.95,1/0.95]
		# 	'elastic_alpha':50, #50   20-100
		# 	'elastic_sigma':7, #7  #kotrolira koliko so maske valovite okoli robov
		# 	'prob_landmark_reduction':0.0, #0.25
		# 	'prob_transform_src':0.0, #0.5
		# }
		self.sb_params1={
			'hull_type':0, # can be 0,1,2,3 (dfl_full,extended,components,facehull)
			'blend_ratio': 0.5,  #[0.25,0.5,0.75,1,1,1]
			'mask_size_h':224+16,
			'mask_size_w':224+16,
			'kernel_1_size':11,
			'kernel_2_size': 11,
			'x_offset':-0.03/2, #from (-0.03,0.03)
			'y_offset':-0.015/2, #from (-0.015,0.015)
			'scale':0.96, #from [0.95,1/0.95]
			'elastic_alpha':45,
			'elastic_sigma':6,
			'prob_landmark_reduction':0.0,
			'prob_transform_src':0.0, #0.5
		}
		

		

		# print(self.sb_params)



	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			# try:
			filename=self.image_list[idx]
			img=np.array(Image.open(filename))
			landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
			# print(landmark.shape)
			bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
			bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
			iou_max=-1
			for i in range(len(bboxes)):
				iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
				if iou_max<iou:
					bbox=bboxes[i]
					iou_max=iou

			landmark=self.reorder_landmark(landmark)

			# do not flip
			# if self.phase=='train':
			# 	if np.random.rand()<0.5:
			# 		img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
			
			img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=False)

			# breakpoint()
			# img_r22,img_f22,mask_f22=self.self_blending(img.copy(),landmark.copy())
			img_r,img_f,mask_f=self.controlled_self_blending(img.copy(),landmark.copy(),self.sb_params0)
			img_r1,img_f1,mask_f1=self.controlled_self_blending(img.copy(),landmark.copy(),self.sb_params1)

			#DO NOT AUGMENT
			# if self.phase=='train':
			# 	transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
			# 	img_f=transformed['image']
			# 	img_r=transformed['image1']
			# 	img_f1=transformed['image']
			# 	img_r1=transformed['image1']

			


			
				
			
			img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=controlled_crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
			img_f1,_,__,___,y0_new,y1_new,x0_new,x1_new=controlled_crop_face(img_f1,landmark,bbox,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
			
			img_r=img_r[y0_new:y1_new,x0_new:x1_new]
			img_r1=img_r1[y0_new:y1_new,x0_new:x1_new]


			
			
			# img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_f1=cv2.resize(img_f1,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
			# img_r1=cv2.resize(img_r1,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255


			

			# img_f=img_f.transpose((2,0,1))
			# img_r=img_r.transpose((2,0,1))
			# img_f1=img_f1.transpose((2,0,1))
			# img_r1=img_r1.transpose((2,0,1))

			if self.my_transform is not None:
				# frame = self.transform(image=frame)["image"]
                # pertrubated_frame = self.transform(image=pertrubated_frame)["image"]
				# print("img_f.shape: ", img_f.shape)
				img_f=self.my_transform(image=img_f.astype('float32'))['image']
				img_r=self.my_transform(image=img_r.astype('float32'))['image']
				img_f1=self.my_transform(image=img_f1.astype('float32'))['image']
				img_r1=self.my_transform(image=img_r1.astype('float32'))['image']

				#duplicate mask dims to get 3 channels instead of 1
				#img_f.shape:  (557, 446, 3)
				#mask_f.shape:  (557, 452, 1)\
				mask_f = np.dstack([mask_f.squeeze()] * 3)
				mask_f1 = np.dstack([mask_f1.squeeze()] * 3)

				#multiply by 255
				mask_f = mask_f * 255
				mask_f1 = mask_f1 * 255

				
				

				# print("mask_f.shape: ", mask_f.shape)
				# #mask_f.shape:  (557, 452, 3, 1)

				# # #remove last dim
				# # mask_f=mask_f[:,:,0]
				# # mask_f1=mask_f1[:,:,0]

				# print("mask_f.shape: ", mask_f.shape)
				

				mask_f=self.my_transform(image=mask_f.astype('float32'))['image']
				mask_f1=self.my_transform(image=mask_f1.astype('float32'))['image']







			flag=False
			# except Exception as e:
			# 	print("Error: ", e)
			# 	idx=torch.randint(low=0,high=len(self),size=(1,)).item()

			#print typr
			# print(type(img_r), type(img_f), type(img_f1))
		
		return img_f,img_f1,img_r,self.label_list[idx],self.image_list[idx],mask_f,mask_f1

	
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask
	

	def controlled_affine(self,img,mask,x_offset=0.03,y_offset=0.015,scale=0.95,elastic_alpha=50,elastic_sigma=7):
		f=alb.Affine(
				translate_percent={'x':x_offset,'y':y_offset},
				scale=scale,
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=elastic_alpha,
				sigma=elastic_sigma,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		# print("mask shape: ", mask.shape)
		return img,mask

	def controlled_self_blending(self,img,landmark,params):

		hull_type=params['hull_type'] # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		blend_ratio=params['blend_ratio']  #[0.25,0.5,0.75,1,1,1]
		mask_size_h=params['mask_size_h'] #(192,257)
		mask_size_w=params['mask_size_w'] #(192,257)
		kernel_1_size=params['kernel_1_size'] #(11,11)
		kernel_2_size=params['kernel_2_size'] #(11,11)
		x_offset=params['x_offset'] #from (-0.03,0.03)
		y_offset=params['y_offset'] #from (-0.015,0.015)
		scale=params['scale'] #from [0.95,1/0.95]
		elastic_alpha=params['elastic_alpha'] #50
		elastic_sigma=params['elastic_sigma'] #7

		prob_landmark_reduction=params['prob_landmark_reduction'] #0.25
		prob_transform_src=params['prob_transform_src'] #0.5


		H,W=len(img),len(img[0])
		if np.random.rand()<prob_landmark_reduction:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			# mask=random_get_hull(landmark,img)[:,:,0]
			mask=get_hull_controlled(landmark,img,hull_type=hull_type)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

	
		source = img.copy()
		# if np.random.rand()<prob_transform_src:
		# 	source = self.source_transforms(image=source.astype(np.uint8))['image']
		# else:
		# 	img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.controlled_affine(source,mask,x_offset=x_offset,y_offset=y_offset,scale=scale,elastic_alpha=elastic_alpha,elastic_sigma=elastic_sigma)
		
	
		img_blended,mask=B.controlled_blend(source,img,mask,blend_ratio=blend_ratio,mask_size_h=mask_size_h,mask_size_w=mask_size_w, kernel_1_size=kernel_1_size, kernel_2_size=kernel_2_size)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

	

		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

	

		source, mask = self.randaffine(source,mask)

		
	
		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_f1,img_r,labels,names=zip(*batch)
		data={}
		# print(img_f[0][0])
		# print(img_f[0][1])
		#important keep the order!!!!
		#torch tensor is used to transform tuples given by zip to tensors
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float(),torch.tensor(img_f1).float()],0)
		# print(type(img_r), type(img_f), type(img_f1))

		# print(isinstance(img_r, torch.Tensor), isinstance(img_f, torch.Tensor), isinstance(img_f1, torch.Tensor))
		# data['img']=torch.cat([img_r,img_f,img_f1],0)
		# data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		data['label']=torch.tensor(labels)

		print("data['img'].shape: ", data['img'].shape)
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

class SBI_Dataset(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8):
		
		assert phase in ['train','val','test']
		
		image_list,label_list=init_ff(phase,'frame',n_frames=n_frames)
		
		path_lm='/landmarks/' 
		label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()


	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				
				filename=self.image_list[idx]
				img=np.array(Image.open(filename))
				landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
				bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
				bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
				iou_max=-1
				for i in range(len(bboxes)):
					iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
					if iou_max<iou:
						bbox=bboxes[i]
						iou_max=iou

				landmark=self.reorder_landmark(landmark)
				if self.phase=='train':
					if np.random.rand()<0.5:
						img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
						
				img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

				img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())

				if self.phase=='train':
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']
					
				
				img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				
				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				

				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))
				flag=False
			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img_f,img_r

	
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask

	def controlled_affine(self,img,mask,x_offset=0.03,y_offset=0.015,scale=0.95,elastic_alpha=50,elastic_sigma=7):
		f=alb.Affine(
				translate_percent={'x':x_offset,'y':y_offset},
				scale=scale,
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=elastic_alpha,
				sigma=elastic_sigma,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask

	def controlled_self_blending(self,img,landmark,params):

		hull_type=params['hull_type'] # can be 0,1,2,3 (dfl_full,extended,components,facehull)
		blend_ratio=params['blend_ratio']  #[0.25,0.5,0.75,1,1,1]
		mask_size_h=params['mask_size_h'] #(192,257)
		mask_size_w=params['mask_size_w'] #(192,257)
		x_offset=params['x_offset'] #from (-0.03,0.03)
		y_offset=params['y_offset'] #from (-0.015,0.015)
		scale=params['scale'] #from [0.95,1/0.95]
		elastic_alpha=params['elastic_alpha'] #50
		elastic_sigma=params['elastic_sigma'] #7

		prob_landmark_reduction=params['prob_landmark_reduction'] #0.25
		prob_transform_src=params['prob_transform_src'] #0.5


		H,W=len(img),len(img[0])
		if np.random.rand()<prob_landmark_reduction:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			# mask=random_get_hull(landmark,img)[:,:,0]
			mask=get_hull_controlled(landmark,img,hull_type=hull_type)
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


		source = img.copy()
		if np.random.rand()<prob_transform_src:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.controlled_affine(source,mask,x_offset=x_offset,y_offset=y_offset,scale=scale,elastic_alpha=elastic_alpha,elastic_sigma=elastic_sigma)

		img_blended,mask=B.controlled_blend(source,img,mask,blend_ratio=blend_ratio,mask_size_h=mask_size_h,mask_size_w=mask_size_w)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_r=zip(*batch)
		data={}
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
		data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
	import blend as B
	from initialize import *
	from funcs import IoUfrom2bboxes,crop_face,RandomDownScale,controlled_crop_face
	import albumentations as alb
	if exist_bi:
		from library.bi_online_generation import random_get_hull,get_hull_controlled
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	def vra_fake_test():
		# strenght = 0.0
		trst = np.linspace(0.0, 1.0, 11)

		imgss=[]
		masks=[]
		difs=[]

		for i in trst:
			strenght = i
			print("strenght: ", strenght)

			test_transform = alb.Compose([
				alb.Resize(384, 384),
				alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],max_pixel_value=255.0),
				ToTensorV2(),
			])
			image_dataset=SBI_Dataset_VRA(phase='trainval',image_size=384,my_transform=test_transform,strength=strenght)
			batch_size=16
			dataloader = torch.utils.data.DataLoader(image_dataset,
							batch_size=batch_size,
							shuffle=False,
							# collate_fn=image_dataset.collate_fn,
							num_workers=0,
							worker_init_fn=image_dataset.worker_init_fn
							)
			data_iter=iter(dataloader)
			# data=next(data_iter)
			# img=data['img']
			# label=data['label']
			# print(img.shape)
			# print(label.shape)
			# print(label)

			img,img1,og,label,name,mask_f,mask_f1=next(data_iter)
			img,img1,og,label,name,mask_f,mask_f1=next(data_iter)
			# img=data
			# label=data['label']
			print(img.shape)
			print(label.shape)
			print(label)



			diff_og_img = torch.abs(og - img)

			img=img.view((-1,3,384,384))
			mask_f=mask_f.view((-1,3,384,384))
			diff_og_img=diff_og_img.view((-1,3,384,384))
			imgss.append(img)
			masks.append(mask_f)
			difs.append(diff_og_img)



			# #concat images
			# img=torch.cat([og,img,diff_og_img,mask_f],0)
			# print("names",len(name))
			# print("names",name)
			# print("labels",label.shape)
			# print("labels",label)
			
			# img=img.view((-1,3,384,384))

			
			
			# img=img.view((-1,3,256,256))
			

			#stacked
			# img=torch.cat([img,img1],1)
			# utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
			# utils.save_image(img, 'loader.png', nrow=batch_size, normalize=True)


		#concat images
		img=torch.cat(imgss,0)
		dif=torch.cat(difs,0)
		mask=torch.cat(masks,0)

		print("img.shape: ", img.shape)
		#save
		utils.save_image(img, 'loader.png', nrow=batch_size, normalize=True)
		utils.save_image(dif, 'loader_dif.png', nrow=batch_size, normalize=True)
		utils.save_image(mask, 'loader_mask.png', nrow=batch_size, normalize=True)

	def ff_real_test():
			test_transform = alb.Compose([
					alb.Resize(384, 384),
					alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],max_pixel_value=255.0),
					ToTensorV2(),
				])
		

			train_dataset=SBI_Dataset_VRA3_real(phase='trainval',my_transform=test_transform)
			batch_size=8
			dataloader=torch.utils.data.DataLoader(train_dataset,
								batch_size=batch_size,
								shuffle=False,
								# collate_fn=train_dataset.collate_fn,
								num_workers=0,
								pin_memory=True,
								drop_last=False,
								worker_init_fn=train_dataset.worker_init_fn,
								)
			


			data_iter=iter(dataloader)
			# img_f,img_r,self.label_list[idx],self.image_list[idx],mask_f,strength,sb_params
			img_f,img_r,label,name,mask_f,strength,sb_params=next(data_iter)
			# label=data['label']
			print(img_f.shape)
			print(label.shape)
			print(label)

			#concat images
			img=torch.cat([img_r,img_f,mask_f],0)
			print("names",len(name))
			print("names",name)
			print("labels",label.shape)
			print("labels",label)
			
			img=img.view((-1,3,384,384))


			print(img.shape)
			

			#stacked
			# img=torch.cat([img,img1],1)
			# utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
			utils.save_image(img, 'loader.png', nrow=batch_size, normalize=True)
		
	
	ff_real_test()


else:
	from utils import blend as B
	from .initialize import *
	from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale,controlled_crop_face
	if exist_bi:
		from utils.library.bi_online_generation import random_get_hull,get_hull_controlled