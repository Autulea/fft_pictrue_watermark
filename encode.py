# coding=utf-8
import cv2
import numpy as np
import os,sys
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import resize
debug = 1
ALPHA = 20

def main():
    global ALPHA,debug
    input = '.\\encode_input.jpg'
    wm = '.\\watermark.png'
    out = '.\\encode_output.jpg'
    alpha = ALPHA
    if not os.path.isfile(input):
        print("image %s does not exist." % input)
    if not os.path.isfile(wm):
        print("watermark %s does not exist." % wm)
    encode(input, wm, out, alpha)


def encode(in_path, wm_path, out_path, alpha):
    global debug
    in_BGR = cv2.imread(in_path)##BGR
    in_height, in_width = in_BGR.shape[0],in_BGR.shape[1]
    wm_mono = cv2.imread(wm_path,0)
    wm_height, wm_width = wm_mono.shape[0],wm_mono.shape[1]
    wm_mono = cv2.resize(wm_mono,(int(in_width/2),int(in_height/2)))
    wm_height, wm_width = wm_mono.shape[0],wm_mono.shape[1]
    out_channal_BGR = cv2.split(in_BGR)
    for i in range(3):
        channal = cv2.split(in_BGR)[i]
        channal_fft = np.fft.fft2(channal)
        channal_fshift = np.fft.fftshift(channal_fft)
        for y in range(0,int(in_height/2)):
            for x in range(0,int(in_width/2)):
                channal_fshift[y][x] =( 1 - wm_mono[y][x]*alpha/254 ) * channal_fshift[y][x]
        channal_fshift_log = np.log(np.abs(channal_fshift))
        plt.imshow(channal_fshift_log,'gray')
        channal_ishift = np.fft.ifftshift(channal_fshift)
        channal_ifft = np.fft.ifft2(channal_ishift)
        channal_ifft = np.abs(channal_ifft)
        out_channal_BGR[i] = channal_ifft
    out_BGR = cv2.merge(out_channal_BGR)
    cv2.imwrite(out_path, out_BGR)
    plt.show()



if __name__ == '__main__':
        main()
