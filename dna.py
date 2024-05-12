import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from importlib import reload  
from bisect import bisect_left as bsearch
import key_generation
import decrypt
import rgb
import encode
import chaos 
import PIL
from PIL import Image
''' 
GLOBAL Constants
'''
# Lorenz paramters and initial conditions
a, b, c = 10, 2.667, 28
x0, y0, z0 = 0, 0, 0

#DNA-Encoding RULE #1 A = 00, T=01, G=10, C=11
dna={}
dna["00"]="A"
dna["01"]="T"
dna["10"]="G"
dna["11"]="C"
dna["A"]=[0,0]
dna["T"]=[0,1]
dna["G"]=[1,0]
dna["C"]=[1,1]
#DNA xor
dna["AA"]=dna["TT"]=dna["GG"]=dna["CC"]="A"
dna["AG"]=dna["GA"]=dna["TC"]=dna["CT"]="G"
dna["AC"]=dna["CA"]=dna["GT"]=dna["TG"]="C"
dna["AT"]=dna["TA"]=dna["CG"]=dna["GC"]="T"
# Maximum time point and total number of time points
tmax, N = 100, 10000

def lorenz(X, t, a, b, c):
    x, y, z = X
    x_dot = -a*(x - y)
    y_dot = c*x - y - x*z
    z_dot = -b*z + x*y
    return x_dot, y_dot, z_dot

def image_selector():                           #returns path to selected image
    path = "NULL"
    root = tk.Tk()
    root.withdraw()                             # we don't want a full GUI, so keep the root window from appearing
    path = filedialog.askopenfilename()         # show an "Open" dialog box and return the path to the selected file
    if path!="NULL":
        print("Image loaded!") 
    else:
        print("Error Image not loaded!")
    return path   

def decompose_matrix(iname):
    image = cv2.imread(iname)
    blue,green,red = rgb.split_into_rgb_channels(image)
    for values, channel in zip((red, green, blue), (2,1,0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype = np.uint8)
        img[:,:] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)
    return B,G,R

def key_matrix_encode(key,b):    
    #encoded key matrix
    b = np.unpackbits(b,axis=1)
    m,n = b.shape
    key_bin = bin(int(key, 16))[2:].zfill(256)
    Mk = np.zeros((m,n),dtype=np.uint8)
    x=0
    for j in range(0,m):
            for i in range(0,n):
                Mk[j,i]=key_bin[x%256]
                x+=1
    
    Mk_enc=np.chararray((m,int(n/2)))
    idx=0
    for j in range(0,m):
        for i in range(0,n,2):
            if idx==(n/2):
                idx=0
            Mk_enc[j,idx]=dna["{0}{1}".format(Mk[j,i],Mk[j,i+1])]
            idx+=1
    Mk_enc=Mk_enc.astype(str)
    return Mk_enc

def xor_operation(b,g,r,mk):
    m,n = b.shape
    bx=np.chararray((m,n))
    gx=np.chararray((m,n))
    rx=np.chararray((m,n))
    b=b.astype(str)
    g=g.astype(str)
    r=r.astype(str)
    for i in range(0,m):
        for j in range (0,n):
            bx[i,j] = dna["{0}{1}".format(b[i,j],mk[i,j])]
            gx[i,j] = dna["{0}{1}".format(g[i,j],mk[i,j])]
            rx[i,j] = dna["{0}{1}".format(r[i,j],mk[i,j])]
         
    bx=bx.astype(str)
    gx=gx.astype(str)
    rx=rx.astype(str)
    return bx,gx,rx 

def plot(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    s = 100
    c = np.linspace(0,1,N)
    for i in range(0,N-s,s):
        ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=(1-c[i],c[i],1), alpha=0.4)
    ax.set_axis_off()
    plt.show()

def sequence_indexing(x,y,z):
    n=len(x)
    fx=np.zeros((n),dtype=np.uint32)
    fy=np.zeros((n),dtype=np.uint32)
    fz=np.zeros((n),dtype=np.uint32)
    seq=sorted(x)
    for k1 in range(0,n):
            t = x[k1]
            k2 = bsearch(seq, t)
            fx[k1]=k2
    seq=sorted(y)
    for k1 in range(0,n):
            t = y[k1]
            k2 = bsearch(seq, t)
            fy[k1]=k2
    seq=sorted(z)
    for k1 in range(0,n):
            t = z[k1]
            k2 = bsearch(seq, t)
            fz[k1]=k2
    return fx,fy,fz
        
def scramble(fx,fy,fz,b,r,g):
    p,q=b.shape
    size = p*q
    bx=b.reshape(size).astype(str)
    gx=g.reshape(size).astype(str)
    rx=r.reshape(size).astype(str)
    bx_s=np.chararray((size))
    gx_s=np.chararray((size))
    rx_s=np.chararray((size))

    for i in range(size):
            idx = fz[i]
            bx_s[i] = bx[idx]
    for i in range(size):
            idx = fy[i]
            gx_s[i] = gx[idx]
    for i in range(size):
            idx = fx[i]
            rx_s[i] = rx[idx]     
    bx_s=bx_s.astype(str)
    gx_s=gx_s.astype(str)
    rx_s=rx_s.astype(str)
    
    b_s=np.chararray((p,q))
    g_s=np.chararray((p,q))
    r_s=np.chararray((p,q))

    b_s=bx_s.reshape(p,q)
    g_s=gx_s.reshape(p,q)
    r_s=rx_s.reshape(p,q)
    return b_s,g_s,r_s

def scramble_new(fx,fy,fz,b,g,r):
    p,q=b.shape
    size = p*q
    bx=b.reshape(size)
    gx=g.reshape(size)
    rx=r.reshape(size)

    bx_s=b.reshape(size)
    gx_s=g.reshape(size)
    rx_s=r.reshape(size)
    
    bx=bx.astype(str)
    gx=gx.astype(str)
    rx=rx.astype(str)
    bx_s=bx_s.astype(str)
    gx_s=gx_s.astype(str)
    rx_s=rx_s.astype(str)
    
    for i in range(size):
            idx = fz[i]
            bx_s[idx] = bx[i]
    for i in range(size):
            idx = fy[i]
            gx_s[idx] = gx[i]
    for i in range(size):
            idx = fx[i]
            rx_s[idx] = rx[i]    

    b_s=np.chararray((p,q))
    g_s=np.chararray((p,q))
    r_s=np.chararray((p,q))

    b_s=bx_s.reshape(p,q)
    g_s=gx_s.reshape(p,q)
    r_s=rx_s.reshape(p,q)

    return b_s,g_s,r_s

def dna_decode(b,g,r):
    m,n = b.shape
    r_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
    g_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
    b_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
    for color,dec in zip((b,g,r),(b_dec,g_dec,r_dec)):
        for j in range(0,m):
            for i in range(0,n):
                dec[j,2*i]=dna["{0}".format(color[j,i])][0]
                dec[j,2*i+1]=dna["{0}".format(color[j,i])][1]
    b_dec=(np.packbits(b_dec,axis=-1))
    g_dec=(np.packbits(g_dec,axis=-1))
    r_dec=(np.packbits(r_dec,axis=-1))
    return b_dec,g_dec,r_dec

def xor_operation_new(b,g,r,mk):
    m,n = b.shape
    bx=np.chararray((m,n))
    gx=np.chararray((m,n))
    rx=np.chararray((m,n))
    b=b.astype(str)
    g=g.astype(str)
    r=r.astype(str)
    for i in range(0,m):
        for j in range (0,n):
            bx[i,j] = dna["{0}{1}".format(b[i,j],mk[i,j])]
            gx[i,j] = dna["{0}{1}".format(g[i,j],mk[i,j])]
            rx[i,j] = dna["{0}{1}".format(r[i,j],mk[i,j])]
         
    bx=bx.astype(str)
    gx=gx.astype(str)
    rx=rx.astype(str)
    return bx,gx,rx 

def recover_image(b,g,r,iname):
    img = cv2.imread(iname)
    img[:,:,2] = r
    img[:,:,1] = g
    img[:,:,0] = b
    cv2.imwrite(("enc_.jpg"), img)
    print("saved ecrypted image as enc.jpg")
    return img    

#program exec9
if (__name__ == "__main__"):
    file_path = image_selector()
    print(file_path)
    key,m,n = key_generation.securekey(file_path)
    x0, y0, z0 = key_generation.update_lorentz(key)
    blue,green,red=decompose_matrix(file_path)
    blue_e,green_e,red_e=encode.dna_encode(blue,green,red)
    Mk_e = key_matrix_encode(key,blue)
    blue_final, green_final, red_final = xor_operation(blue_e,green_e,red_e,Mk_e)
    x,y,z=chaos.gen_chaos_seq(m,n,x0,y0,z0,a, b, c)
    fx,fy,fz=sequence_indexing(x,y,z)
    blue_scrambled,green_scrambled,red_scrambled = scramble(fx,fy,fz,blue_final,red_final,green_final)
    b,g,r=dna_decode(blue_scrambled,green_scrambled,red_scrambled)
    img=recover_image(b,g,r,file_path)
    print("decrypting...")
    #img = PIL.Image.open(r"C:\crypto\enc_.jpg")
    file_path = image_selector()
    img = image = cv2.imread(file_path)
    decrypt.decrypt(img,fx,fy,fz,file_path,Mk_e,blue,green,red)