import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def xy_gradients(img):
	#get gradients in x and y directions
	grad_x = cv2.Sobel(img,cv2.CV_64F, 1 , 0, ksize = 3)
	grad_y = cv2.Sobel(img,cv2.CV_64F, 0 , 1, ksize = 3)
	#get magnitude and angle of the gradient
	grad_mag = np.sqrt((grad_x**2 + grad_y**2))

	#normalise grad_x and grad_y
	grad_x = grad_x/grad_mag
	grad_y = grad_y/grad_mag
	
	grad_x[np.isnan(grad_x)] = 0
	grad_y[np.isnan(grad_y)] = 0

	return grad_x,grad_y


def get_rays(img,theta_slack,direction):
	edges = cv2.Canny(img,100,200)
	grad_x,grad_y = xy_gradients(img)
	rays = []	
	#NOTE : the x coordinate moves according to column no and y coordinate = row no.
	for y in range(edges.shape[0]):
		for x in range(edges.shape[1]):
			if(edges[y,x]>0):
				dx = direction*grad_x[y,x]
				dy = direction*grad_y[y,x]
				if(dx == 0 and dy == 0):
					continue
				prev_x , prev_y = x , y
				i = 1
				ray = []
				ray.append((x,y))
				while(1):
					cur_x = int(np.floor(x + dx*i))
					cur_y = int(np.floor(y + dy*i))
					i+=1
					# if we are still at the old pixel
					if((cur_x == prev_x) and (cur_y==prev_y)):
						continue
					# if we have exceeded image boundaries
					# we dont add rays that end at image boundaries(all rays must end at another edge) 
					if((cur_x>=edges.shape[1]) or (cur_y>=edges.shape[0]) or cur_x<0 or cur_y<0):
						break	
					ray.append((cur_x,cur_y))
					prev_x = cur_x
					prev_y = cur_y
					if(edges[cur_y,cur_x] == 0):
						continue
					# if we are at another edge
					# since grad_ x and grad_y are normalised , grad_x = cos(angle) , grad_y = sin(angle)
					# Using formula of cos(A-B) to check if A-B is within threshold
					
					temp = grad_x[y,x] * -grad_x[cur_y, cur_x] + grad_y[y,x] * -grad_y[cur_y, cur_x]

					if( np.arccos(temp) < theta_slack):
						rays.append(ray)
					# always break at an edge irrespective of whether ray is valid or not 
					break	
	return rays

def dist(p,q):
	return ((p[0]-q[0])**2 + (p[1] - q[1])**2)**0.5


def swt(img,theta_slack = np.pi/4,moderation = "median" , direction = 1):
	 
	rays = get_rays(img,theta_slack,direction)
	swt = float('inf')*np.ones(img.shape)

	#1st pass for swt formation
	for ray in rays:
		width = dist(ray[0],ray[-1])
		for p in ray:
			if(width<swt[p[1],p[0]]):
				swt[p[1],p[0]] = width
				
	#2nd pass : replace with mode or median of each ray 
	for ray in rays:
		temp_list = [swt[p[1],p[0]] for p in ray]


		if moderation is "median":
			replace_val = np.median(temp_list)
		else:
			replace_val = stats.mode(temp_list).mode[0]
		
		for p in ray:
			swt[p[1],p[0]] = min(swt[p[1],p[0]],replace_val)

	return swt		



















