import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tf
import scipy.ndimage as ni
from scipy.fft import fft,rfft
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.sparse import issparse
from skimage.measure import block_reduce
from skimage.morphology import skeletonize
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import gaussian as im_gaussian

def read_avgs(experiment_dir,codebase='kaan',downsample=0,downsample_func=np.mean):
    avg_mov_dict = {}
    with tqdm(total=len(os.listdir(experiment_dir))) as pbar:
        if codebase == 'kaan':
            for exp in os.listdir(experiment_dir):
                if os.path.isdir(pjoin(experiment_dir,exp)):
                    key = exp.split('_')[-1]
                    pbar.set_description(key)
    
                    try:
                        # for ISI
                        avg_fname = [f for f in os.listdir(pjoin(experiment_dir,exp,'movies')) if f.startswith('inv_avg')][0]
                    except:
                        # 1P GCaMP
                        avg_fname = [f for f in os.listdir(pjoin(experiment_dir,exp,'movies')) if f.startswith('avg')][0]

                    avg_file = pjoin(experiment_dir,exp,'movies',avg_fname)
                    
                    temp_mov = tf.imread(avg_file)

                    try:
                        temp_mov = temp_mov[:,0,:,:]
                    except:
                        pass
                    
                    temp_mov = temp_mov[:,:,::-1]

                    if downsample:
                        avg_mov_dict[key] = downsample_movie(temp_mov,block_size=downsample,func=downsample_func)
                    else:
                        avg_mov_dict[key] = temp_mov
                        
                    pbar.update()
                    
        elif codebase == 'ben':
            for exp in os.listdir(experiment_dir):
                if os.path.isdir(pjoin(experiment_dir,exp)):
                    key_exp = exp.split('_')[-1]
                    pbar.set_description(key_exp)
                    for avg_fname in os.listdir(pjoin(experiment_dir,exp,'movies')):
                        if 'avg' in avg_fname:
                            key = f'{key_exp}_{avg_fname[:-4]}'
                            avg_file = pjoin(experiment_dir,exp,'movies',avg_fname)
                            temp_mov = tf.imread(avg_file)
                            try:
                                temp_mov = temp_mov[:,0,:,:]
                            except:
                                pass

                            temp_mov = temp_mov[:,:,::-1]
                            if downsample:
                                avg_mov_dict[key] = downsample_movie(temp_mov,block_size=downsample,func=downsample_func)
                            else:
                                avg_mov_dict[key] = temp_mov
                    pbar.update()
    return avg_mov_dict

def downsample_image(image,block_size=2,func=np.mean):
    """ Downsamples a given image by the block_size"""
    if not isinstance(block_size,list):
        block_size = (block_size,block_size)
    else:
        assert block_size[0]==block_size[1], f'Only square kernels allowed for downsapling'
        
    if len(image.shape)>2:
        if image.shape[0] == 1:
            # a 2d image in a 3d matrix shape
            image = image[0,:,:]
        else:
            raise ValueError(f"Input image can't have a 3rd dimension bigger than 1!")
    elif len(image.shape)==2:
        pass
    else:
        raise ValueError(f"Input image is not 2D!!")
    
    return block_reduce(image,block_size=block_size,func=func)

def downsample_movie(movie,block_size=2,func=np.mean):
    """Wrapper for downsampling whole movies """
    if len(movie.shape) < 3:
        raise ValueError(f'Shape of movie input is wrong: {len(movie.shape)}<3')

    new_shape = tuple([int(i/block_size) for i in movie.shape])
    new_movie = np.zeros((movie.shape[0],new_shape[1],new_shape[2]))
    for i in range(movie.shape[0]):
        new_movie[i,:,:] = downsample_image(movie[i,:,:],block_size=block_size,func=func)

    return new_movie

def fft_movie(movie, component = 1,output_raw = False):
    '''
    Computes the fft of a movie and returns the magnitude and phase 
    '''
    movief = fft(movie, axis = 0)
    if output_raw:
        return movief[component]
    phase  = -1. * np.angle(movief[component]) % (2*np.pi)
    mag = (np.abs(movief[component])*2.)/len(movie)
    return mag,phase

def phasemag_to_hsv(phase,mag,startdeg=0,stopdeg=140,vperc=99,sperc=90):
    H = phasemap_to_visual_degrees(phase,startdeg,stopdeg)
    H /= np.max(H)
    V = mag.copy()
    V /= np.percentile(mag,vperc)
    S = mag**0.3
    S /= np.percentile(S,sperc)
    # Normalization for opencv ranges 0-255 for uint8
    hsvimg = np.clip(np.stack([H,S,V],axis=2).astype(np.float32),0,1)
    hsvimg *= 255
    return cv2.cvtColor(hsvimg.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)

def hsv_colorbar(col_range=[10,190]):
    hsvimg = np.ones([1,255,3])*255
    hsvimg[:,:,0] = np.linspace(col_range[0],col_range[1],255)
    hsvimg = hsvimg.transpose([1,0,2])
    return cv2.cvtColor(hsvimg.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)

def phasemap_to_visual_degrees(phasemap,startdeg,stopdeg):
    '''
    Normalizes the phasemap to visual angles
    Joao Couto 2019
    '''
    res = phasemap.copy() - np.nanmin(phasemap)
    res /= np.nanmax(res)
    res *= np.abs(np.diff([startdeg,stopdeg]))
    res += startdeg
    return res

def visual_sign_map(phasemap1, phasemap2):
    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    graddir2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i,j] = np.arctan2(gradmap1[1][i,j],gradmap1[0][i,j])
            graddir2[i,j] = np.arctan2(gradmap2[1][i,j],gradmap2[0][i,j])

    vdiff = np.multiply(np.exp(1j * graddir1),np.exp(-1j * graddir2))
    areamap = np.sin(np.angle(vdiff))
    return areamap

def _handle_sparse(im,shape):
    if issparse(im):
        if shape is None:
            raise ValueError('Supply shape = [H,W] when using sparse arrays')
        im = np.asarray(im.todense()).reshape(shape)
    return im

def nb_play_movie(data,interval=30,shape = None,**kwargs):
    ''' 
    Play a movie from the notebook
    '''
    from ipywidgets import Play,jslink,HBox,IntSlider
    from IPython.display import display

    i = _handle_sparse(data[0],shape = shape)
    im = plt.imshow(i.squeeze(),**kwargs)
    slider = IntSlider(0,min = 0,max = data.shape[0]-1,step = 1,description='Frame')
    play = Play(interval=interval,
                value=0,
                min=0,
                max=data.shape[0]-1,
                step=1,
                description="Press play",
                disabled=False)
    jslink((play, 'value'), (slider, 'value'))
    display(HBox([play, slider]))
    def updateImage(change):
        i = _handle_sparse(data[change['new']],shape=shape)
        im.set_data(i.squeeze())
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
    slider.observe(updateImage, names='value')
    return dict(fig = plt.gcf(),
                ax=plt.gca(),
                im= im,
                update = updateImage)
    
def parseCamLog(fname):
    """
    Parses the camlog
    """
    comments = []
    with open(fname,'r') as fd:
        for i,line in enumerate(fd):
            if line.startswith('#'):
                comments.append(line.strip('\n').strip('\r'))
    
    commit = None
    for c in comments:
        if c.startswith('# Log header:'):
            cod = c.strip('# Log header:').strip(' ').split(',')
            camlogheader = [c.replace(' ','') for c in cod]
        elif c.startswith('# Commit hash:'):
            commit = c.strip('# Commit hash:').strip(' ')

    camdata = pd.read_csv(fname,
                      names = camlogheader,
                      delimiter=',',
                      header=None,comment='#',
                      engine='c')
    
    
    return camdata,comments,commit