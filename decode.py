import cv2
import numpy as np
import os,sys
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import resize
debug = 1


def main():
    global ALPHA,debug
    input = '.\\encode_output.jpg'
    if not os.path.isfile(input):
        print("image %s does not exist." % input)
    decode(input)

def decode(in_path):
    global debug
    in_BGR = cv2.imread(in_path,0)
    channal_fft = np.fft.fft2(in_BGR)
    channal_fshift = np.fft.fftshift(channal_fft)
    channal_fshift_log = np.log(np.abs(channal_fshift))
    plt.imshow(channal_fshift_log,'gray')
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
        main()