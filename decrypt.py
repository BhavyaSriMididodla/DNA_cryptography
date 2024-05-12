import rgb
import numpy as np
import cv2
import encode
import dna
import matplotlib.pyplot as plt

def decrypt(image,fx,fy,fz,fp,Mk,bt,gt,rt):
    plt.imshow(image)
    plt.show()
    r,g,b=rgb.split_into_rgb_channels(image)
    p,q = rt.shape
    benc,genc,renc=encode.dna_encode(b,g,r)
    bs,gs,rs=dna.scramble_new(fx,fy,fz,benc,genc,renc)
    bx,rx,gx=dna.xor_operation_new(bs,gs,rs,Mk)
    blue,green,red=encode.dna_decode(bx,gx,rx)
    green,red = red, green
    img=np.zeros((p,q,3),dtype=np.uint8)
    img[:,:,0] = red
    img[:,:,1] = green
    img[:,:,2] = blue
    cv2.imwrite(("Recovered_.jpg"), img)
    print("saved decrypted image as recovered.jpg")