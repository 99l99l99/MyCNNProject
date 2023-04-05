from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
# 自定义数据集类
class BaseDataset(Dataset):
	def __init__(self, cfg_data):
		self.cfg = cfg_data
		self.image_list = []
		self.label_list = []
		self.seg_list = []
		for path in cfg_data['path']:
			image_list = []
			label_list = []
			seg_list = []
			with open(path+"label.txt", 'r') as f:
				lines = f.readlines()
				for line in lines:
					line = line.split(",")
					img_path = path + "car.0.RGB." + line[0] + ".jpeg"
					seg_path = path + "car.0.GRAY." + line[0] + ".png"
					points = [int(i) for i in line[1:]]
					image_list.append(img_path)
					seg_list.append(seg_path)
					label_list.append(points)
			self.image_list += image_list
			self.label_list += label_list
			self.seg_list += seg_list

		assert len(self.image_list) == len(self.label_list) == len(self.seg_list)
		self._mix_list = glob.glob("./data/mix_up/*/*.jpg")

	def __getitem__(self, index):
		raise NotImplementedError

	def __len__(self):
		return len(self.image_list)

	def paste_uav(self, backgroud, size, wh_ratio, paste_rate, cfg, alpha_beta = 9):
		if self.uniform(0,1) < paste_rate:
			i = self.randint(0, len(self.image_list)-1)
			uav = cv2.imread(self.image_list[i])
			seg = cv2.imread(self.seg_list[i], cv2.IMREAD_GRAYSCALE)
			label = torch.tensor(self.label_list[i], dtype = torch.float)
			label = label.view(-1,3)
			label_kind = label[:,0]
			label_coord = label[:,1:]
			uav_size = self.randint(*size)
			min_ratio = min(wh_ratio, 1/wh_ratio)
			max_ratio = 1 / min_ratio
			uav_ratio = self.uniform(min_ratio, max_ratio)
			uav_h = int(uav_size * uav_ratio)
			uav_w = int(uav_size / uav_ratio)
			# print("uav_h: %d, uav_w: %d" % (uav_h, uav_w))

			# uav_w = self.cfg['resize'][0]
			# uav_h = self.cfg['resize'][1]
			uav = cv2.resize(uav, (uav_w, uav_h))
			if cfg['photometric']['enable']:
				uav = self.photometric_augmentation(uav, cfg['photometric']['params'])

			H = np.eye(3, dtype='float64')
			H[0,0] = uav_w / seg.shape[1]
			H[1,1] = uav_h / seg.shape[0]
			seg = cv2.warpPerspective(seg, H, (uav.shape[1], uav.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
			label_coord = self.warpPerspectivePoints(label_coord, H)

			# xcenter = self.randint(0, backgroud.shape[1] - 1)
			# ycenter = self.randint(0, backgroud.shape[0] - 1)
			xcenter = self.betaInt(0, backgroud.shape[1] - 1, alpha_beta, alpha_beta)[0]
			ycenter = self.betaInt(0, backgroud.shape[0] - 1, alpha_beta, alpha_beta)[0]
			# xcenter = int(backgroud.shape[1] / 2)
			# ycenter = int(backgroud.shape[0] / 2)
			xmin = xcenter - uav_w // 2
			ymin = ycenter - uav_h // 2
			xmax = xmin + uav_w
			ymax = ymin + uav_h

			#计算贴图后的关键点坐标，偏差为背景左上角与贴图左上角的偏差，及xmin，ymin
			label_coord[:,0] = label_coord[:,0] + xmin
			label_coord[:,1] = label_coord[:,1] + ymin

			# cut pixel index in uav/seg
			lxmin = max(xmin, 0) - xmin
			lymin = max(ymin, 0) - ymin
			lxmax = min(xmax, backgroud.shape[1]) - xmin
			lymax = min(ymax, backgroud.shape[0]) - ymin

			# cut pixel index in backgroud
			xmin = max(xmin, 0)
			ymin = max(ymin, 0)
			xmax = min(xmax, backgroud.shape[1])
			ymax = min(ymax, backgroud.shape[0])

			uav_mask = (seg[lymin:lymax,lxmin:lxmax] > 0).astype('int32')
			backgroud_part = backgroud[ymin:ymax,xmin:xmax] * (1 - uav_mask[...,np.newaxis])
			uav_part = uav[lymin:lymax,lxmin:lxmax] * uav_mask[...,np.newaxis]

			backgroud[ymin:ymax,xmin:xmax] = backgroud_part + uav_part

			label = np.concatenate([label_kind[:,np.newaxis], label_coord],axis = 1)
		else:
			label = np.zeros((0,3),np.float32)
			uav_size = 0
		return backgroud, label, uav_size

	# points: tensor(4,2) H:np.array(3,3)
	def warpPerspectivePoints(self, points, H):
		points = np.array(points)
		points = np.concatenate((points, np.ones((points.shape[0],1))), axis = 1)
		points = points @ H.T
		return torch.tensor(points[:,0:2])

	def randint(self, low, high, size=None):
		low = int(low)
		high = int(high)
		return np.random.randint(low, high+1, size=size)

	def uniform(self, low, high, size=None):
		return np.random.uniform(low, high, size=size)

	def beta(self, low, high, alpha, beta, size=[1]):
		Beta = torch.distributions.Beta(alpha,beta)
		return (Beta.sample(size) * (high - low) + low).cpu().numpy()

	def betaInt(self, low, high, alpha, beta, size=[1]):
		Beta = torch.distributions.Beta(alpha,beta)
		return (Beta.sample(size) * (high - low) + low).cpu().numpy().astype(np.int)
		
	def rand(self, size=None):
		return np.random.rand(*size) if size is not None else np.random.rand()

	def gaussian(self, mean, std, size=None):
		return np.random.randn(*size) * std + mean if size is not None else np.random.randn() * std + mean
	

	'''
	HomographicAugmentation
	'''

	def homographic_augmentation(self, h, w, config):
		H = np.eye(3, dtype='float64')

		if 'perspective' in config:
			H = self.random_perspective(h, w, H, config['perspective'])
		if 'crop' in config:
			H = self.random_crop(h, w, H, config['crop'])
		if 'pad' in config:
			H = self.random_pad(h, w, H, config['pad'])
		if 'flip' in config:
			H = self.random_flip(h, w, H, config['flip'])
		if 'rotate' in config:
			H = self.random_rotate(h, w, H, config['rotate'])
		if 'yolo' in config:
			H = self.yolo_enhance(h, w, H, config['yolo'])

		return H

	def random_flip(self, h, w, H, config):
		H_ = np.array([[1,0,0,0,1,0,0,0,1]]).reshape(3,3)
		if config:
			if self.randint(0,1) == 0:
				H_[0,0] = -1
				H_[0,2] = w
			#if self.randint(0,1) == 0:
			#	H_[1,1] = -1
			#	H_[1,2] = h
		
		return H_ @ H

	def random_crop(self, h, w, H, config):

		if config['mode'] == 'ratio':
			xmin = self.randint(int(w*config['w_range'][0]), int(w*config['w_range'][1]))
			xmax = self.randint(int(w*(1-config['w_range'][1])), w-1-int(w*config['w_range'][0]))
			ymin = self.randint(int(h*config['h_range'][0]), int(h*config['h_range'][1]))
			ymax = self.randint(int(h*(1-config['h_range'][1])), h-1-int(h*config['h_range'][0]))
		elif config['mode'] == 'pixel':
			xmin = self.randint(int(config['w_range'][0]), int(config['w_range'][1]))
			xmax = self.randint(int(w-config['w_range'][1])), w-1-int(config['w_range'][0])
			ymin = self.randint(int(config['h_range'][0]), int(config['h_range'][1]))
			ymax = self.randint(int(h-config['h_range'][1])), h-1-int(config['h_range'][0])

		else:
			raise ValueError('Unknown crop mode %s' % config['mode'])

		# xmin = max(xmin, 0)
		# xmax = min(xmax, w-1)
		# ymin = max(ymin, 0)
		# ymax = min(ymax, h-1)

		# H_ = np.eye(3, dtype='float64')
		# H_[0,0] = w / (xmax - xmin)
		# H_[0,2] = - xmin * w / (xmax - xmin)
		# H_[1,1] = h / (ymax - ymin)
		# H_[1,2] = - ymin * h / (ymax - ymin)

		H_ = np.eye(3, dtype='float64')
		H_[0,0] = 1
		H_[0,2] = - xmin
		H_[1,1] = 1
		H_[1,2] = - ymin

		return H_ @ H

	def random_pad(self, h, w, H, config):
		
		if config['mode'] == 'ratio':
			xl = self.randint(int(w*config['w_range'][0]), int(w*config['w_range'][1]))
			xr = self.randint(int(w*config['w_range'][0]), int(w*config['w_range'][1]))
			yt = self.randint(int(h*config['h_range'][0]), int(h*config['h_range'][1]))
			yb = self.randint(int(h*config['h_range'][0]), int(h*config['h_range'][1]))
		elif config['mode'] == 'pixel':
			xl = self.randint(int(config['w_range'][0]), int(config['w_range'][1]))
			xr = self.randint(int(config['w_range'][0]), int(config['w_range'][1]))
			yt = self.randint(int(config['h_range'][0]), int(config['h_range'][1]))
			yb = self.randint(int(config['h_range'][0]), int(config['h_range'][1]))

		else:
			raise ValueError('Unknown pad mode %s' % config['mode'])

		xl = max(xl, 0)
		xr = max(xr, 0)
		yt = max(yt, 0)
		yb = max(yb, 0)

		H_ = np.eye(3, dtype='float64')
		H_[0,0] = (w - xl - xr) / (w + xl + xr)
		H_[0,2] = xl * w / (w + xl + xr)
		H_[1,1] = (h - yt - yb) / (h + yt + yb)
		H_[1,2] = yt * h / (h + yt + yb)

		return H_ @ H

	def random_perspective(self, h, w, H, config):
		corner = np.array([[0,0],
						   [0,h],
						   [w,0],
						   [w,h]], dtype='float64')
		distort_corner = self.uniform(*config['corner_range'], size=corner.shape)
		distort_corner = np.minimum(distort_corner, np.array([[w/3, h/3]]))
		distort_corner = np.maximum(distort_corner, -np.array([[w/3, h/3]]))
		distort_corner += corner
		perspectiveMatrix = cv2.getPerspectiveTransform(corner.astype(np.float32), distort_corner.astype(np.float32))
		return perspectiveMatrix @ H


	def random_rotate(self, h, w, H, config):
		r = self.randint(*config['angle'])
		scale = self.uniform(*config['scale'])
		rotateMatrix = cv2.getRotationMatrix2D((w*0.5, h*0.5), r, float(scale)) # 旋转变化矩阵
		rotateMatrix = np.concatenate([rotateMatrix, np.array([[0,0,1]], dtype='float64')], axis=0)
		return rotateMatrix @ H


	def yolo_enhance(self, h, w, H, config):

		new_ar = w/h * self.uniform(1-config['jitter'],1+config['jitter'])/self.uniform(1-config['jitter'],1+config['jitter'])
		# 缩放系数
		scale = self.uniform(config['scale'], 1/config['scale'])
		H_ = np.eye(3, dtype='float64')
		if new_ar < 1:
			nh = int(scale*h)
			nw = int(nh*new_ar)
		else:
			nw = int(scale*w)
			nh = int(nw/new_ar)

		xl = self.randint(0, abs(nw-w))
		xr = abs(nw-w) - xl
		yt = self.randint(0, abs(nh-h))
		yb = abs(nh-h) - yt

		if nw - w > 0:
			H_[0,0] = (w + xr + xl) / (w - xr - xl)
			H_[0,2] = - xl * w / (w - xr - xl)
		else:
			H_[0,0] = (w - xl - xr) / (w + xl + xr)
			H_[0,2] = xl * w / (w + xl + xr)

		if nh - h > 0:
			H_[1,1] = (h + yt + yb) / (h - yt - yb)
			H_[1,2] = - yt * h / (h - yt - yb)
		else:
			H_[1,1] = (h - yt - yb) / (h + yt + yb)
			H_[1,2] = yt * h / (h + yt + yb)

		return H_ @ H


	'''
	PhotometricAugmentation
	'''

	def photometric_augmentation(self, image, config):
		if 'brightness' in config:
			image = self.random_brightness(image, config['brightness'])
		if 'contrast' in config:
			image = self.random_contrast(image, config['contrast'])
		if 'hue_saturation' in config:
			image = self.random_hue_saturation(image, config['hue_saturation'])
		if 'gaussian_noise' in config:
			image = self.additive_gaussian_noise(image, config['gaussian_noise'])
		if 'speckle_noise' in config:
			image = self.additive_speckle_noise(image, config['speckle_noise'])
		if 'shade' in config:
			image = self.additive_shade(image, config['shade'])
		if 'gaussian_blur' in config:
			image = self.gaussian_blur(image, config['gaussian_blur'])
		if 'motion_blur' in config:
			image = self.motion_blur(image, config['motion_blur'])
		if 'stroboscopic' in config:
			image = self.random_stroboscopic(image, config['stroboscopic'])
		if 'mixup' in config:
			image = self.mixup(image, config['mixup'])

		if 'ellipse' in config:
			image = self.draw_ellipses(image, config['ellipse'])

		return image


	def random_brightness(self, image, config):
		if 'max_abs_change' in config:
			change = self.randint(-config['max_abs_change'], config['max_abs_change'])
		else:
			change = self.randint(*config['change_range'])
		image = image.astype('int32') + change
		return np.clip(image, 0, 255).astype('uint8')
		
	
	def random_contrast(self, image, config):
		t = []
		contrast = self.uniform(*config['strength_range'])
		mean = np.mean(image)
		image = mean + (image - mean) * contrast
		return np.clip(image, 0, 255).astype('uint8')

	
	def random_hue_saturation(self, image, config):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		image = image.astype('float32')

		if 'hue_range' in config:
			image[...,0] += self.randint(*config['hue_range'])
			image[...,0] = (image[...,0] + 180)%180
		if 'saturation_range' in config:
			image[...,1] *= self.uniform(*config['saturation_range'])
			image[...,1] = np.minimum(image[...,1],255)

		image = np.round(image)
		image = image.astype('uint8')
		image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

		return image


	def additive_gaussian_noise(self, image, config):
		sigma = config['stddev_range'][0] + self.rand() * config['stddev_range'][1]
		gaussian_noise = self.rand(image.shape) * sigma
		image = image + gaussian_noise
		return np.clip(image, 0, 255).astype(np.uint8)
		

	def additive_speckle_noise(self, image, config):
		intensity = self.randint(0, config['intensity'])
		noise = self.randint(0, 255, image.shape).astype('uint8')
		black = noise < intensity
		white = noise > 255 - intensity
		image[white > 0] = 255
		image[black > 0] = 0
		return image


	def additive_shade(self, image, config):
		transparency = self.uniform(*config['transparency_range'])
		min_dim = min(image.shape) / 4
		mask = np.zeros(image.shape[:2], np.uint8)
		for i in range(config['nb_ellipses']):
			ax = int(max(self.rand() * min_dim, min_dim / 5))
			ay = int(max(self.rand() * min_dim, min_dim / 5))
			max_rad = max(ax, ay)
			x = self.randint(max_rad, image.shape[1] - max_rad)  # center
			y = self.randint(max_rad, image.shape[0] - max_rad)
			angle = self.rand() * 90
			cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

		kernel_size = int( config['kernel_size_range'][0] + self.rand() *
						  (config['kernel_size_range'][1] - config['kernel_size_range'][0]))
		if (kernel_size % 2) == 0:  # kernel_size has to be odd
			kernel_size += 1
		mask = cv2.GaussianBlur(mask.astype(np.float), (kernel_size, kernel_size), 0)
		mask = mask[...,np.newaxis]
		image = image * (1 - transparency * mask/255.)
		return np.clip(image, 0, 255).astype(np.uint8)


	def motion_blur(self, image, config):
		
		ksizex = self.randint(0, config['max_kernel_size_x'])*2 + 1  # make sure is odd
		ksizey = self.randint(0, config['max_kernel_size_x'])*2 + 1  # make sure is odd

		if ksizex * ksizey <= 1:
			return image
		centerx = int((ksizex-1)/2)
		centery = int((ksizey-1)/2)
		kernel = np.zeros((ksizey, ksizex))
		if ksizey > ksizex: # 'h'
			kernel[centery, :] = 1.
		else: # 'v'
			kernel[:, centerx] = 1.
		var = min(ksizex, ksizey) ** 2 / 4.
		gridx = np.arange(ksizex)
		gridy = np.arange(ksizey)
		gridx, gridy = np.meshgrid(gridx, gridy)
		gaussian = np.exp(-(np.square(gridx-centerx)+np.square(gridy-centery))/(2.*var))
		r = self.randint(0,180)
		r = cv2.getRotationMatrix2D((centerx, centery), r, float(1)) # 旋转变化矩阵
		kernel = cv2.warpAffine(kernel, r, (ksizex, ksizey))
		kernel *= gaussian
		kernel /= np.sum(kernel)
		image = cv2.filter2D(image.astype(np.uint8), -1, kernel)
		if len(image.shape) == 2:
			image = image[...,np.newaxis]
		return image

	def gaussian_blur(self, image, config):
		ksize = self.randint(0, config['max_kernel_size']) * 2 + 1
		if ksize == 1:
			return image
		sigma = self.randint(1, config['max_sigma'])
		image = cv2.GaussianBlur(image, ksize=(ksize,ksize), sigmaX=sigma, sigmaY=sigma)
		if len(image.shape) == 2:
			image = image[...,np.newaxis]
		return image

    # 频闪
	def random_stroboscopic(self, image, config):
		gamma_random = self.uniform(0, np.pi*2)
		omega = self.uniform(*config['omega_range'])
		amplitude = self.uniform(*config['amplitude_range'])
		bias = self.uniform(*config['bias_range'])

		image_gamma = np.sin(self.image_*omega+gamma_random) #
		image_gamma[...,0] *= amplitude*2 + bias
		image_gamma[...,1] *= amplitude + bias
		image_gamma[...,2] *= 0

		image = np.round(image + image_gamma)
		image[image>255] = 255
		image[image<0] = 0

		return image

	def mixup(self, image, config):
		mixim = cv2.imread(self._mix_list[self.randint(0,len(self._mix_list)-1)])
		mixim = cv2.resize(mixim, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
		ratio = self.uniform(0, config['max_ratio'])
		image = image.astype('float32')
		mixim = mixim.astype('float32')
		image = (1-ratio) * image + ratio * mixim
		return image.round().astype('uint8')

	'''
	def cutmix(self, image, bbox, config, mask=None):
		if self.uniform(0, 1) > config['probability']:
			return image
		cutim = cv2.imread(self._mix_list[self.randint(0,len(self._mix_list)-1)])
		cutim = cv2.resize(cutim, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
		if mask is None:
			mask = np.zeros_like(image[...,:1])
			for i in range(bbox.shape[0]):
				mask[bbox[i,2]:bbox[i,4],bbox[i,1]:bbox[i,3]] = 1
		cv2.imshow('1', mask * 255)
		return mask * image + (1 - mask) * cutim
	'''

	def cutmix(self, image, mask, config):
		if self.uniform(0, 1) > config['probability']:
			return image
		cutim = cv2.imread(self._mix_list[self.randint(0,len(self._mix_list)-1)])
		cutim = cv2.resize(cutim, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)

		return mask * image + (1 - mask) * cutim