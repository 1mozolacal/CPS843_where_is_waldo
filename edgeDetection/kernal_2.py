#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt

def plti(im, **kwargs):
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off') # turn off axis
    plt.show()  

im_path = 'snip.jpg'#'page.jpg'
im = imageio.v2.imread(str(im_path))


# In[2]:


print(type(imageio.core.asarray(im)))
nArr = imageio.core.asarray(im)
plti(nArr)


# In[3]:


from scipy import signal, misc
#We are looking for thick horizontal lines of red and white that are above and bellow eachother

red_to_white_arr = np.copy(nArr)

waldo_kernal = np.array([
    [[0.5,-0.25,-0.25],[0.5,-0.25,-0.25],[0.5,-0.25,-0.25],[0.5,-0.25,-0.25],[0.5,-0.25,-0.25]],
    [[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33]],
])

waldo_kernal = waldo_kernal/ (waldo_kernal.shape[0] * waldo_kernal.shape[1])

search = red_to_white_arr[:,:,:]

for col in range(3):
    search[:,:,col] = signal.convolve2d(search[:,:,col],waldo_kernal[:,:,col],mode='same')
    
b_and_w = np.sum(search,axis=2)
b_and_w[b_and_w<0] = 0
    
plt.imshow(b_and_w)


# In[4]:


white_to_red_arr = np.copy(nArr)

waldo_kernal = np.array([
    [[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33]],
    [[0.5,-0.25,-0.25],[0.5,-0.25,-0.25],[0.5,-0.25,-0.25],[0.5,-0.25,-0.25],[0.5,-0.25,-0.25]],
])

waldo_kernal = waldo_kernal/ (waldo_kernal.shape[0] * waldo_kernal.shape[1])

search = white_to_red_arr[:,:,:]

for col in range(3):
    search[:,:,col] = signal.convolve2d(search[:,:,col],waldo_kernal[:,:,col],mode='same')
    
b_and_w_2 = np.sum(search,axis=2)
b_and_w_2[b_and_w_2<0] = 0
    
plt.imshow(b_and_w_2)


# In[5]:


combined = np.maximum(b_and_w,b_and_w_2)
plt.imshow(combined)


# In[6]:


#5X5 blur the choice maxium pixel
from scipy.ndimage import gaussian_filter
from numpy import unravel_index

blurred = gaussian_filter(combined,sigma=1)
index = unravel_index(blurred.argmax(), blurred.shape)
print(index)

# Show the image
plt.imshow(blurred)

