import hashlib 
from PIL import Image
import textwrap

def securekey (iname):
    img = Image.open(iname)
    m, n = img.size
    print("pixels: {0}  width: {2} height: {1} ".format(m*n, m, n))
    pix = img.load()          
    plainimage = list()                         #_plainimage contains all the rgb values continuously
    for y in range(n):
        for x in range(m):
            for k in range(0,3):
                plainimage.append(pix[x,y][k])    
    key = hashlib.sha256()                      #key is made a hash.sha256 object  
    key.update(bytearray(plainimage))          #image data is fed to generate digest
    return key.hexdigest() ,m ,n

x0, y0, z0 = 0, 0, 0
def update_lorentz (key):
    key_bin = bin(int(key, 16))[2:].zfill(256)  #covert hex key digest to binary
    k={}                                        #key dictionary
    key_32_parts=textwrap.wrap(key_bin, 8)      #slicing key into 8 parts
    num=1
    for i in key_32_parts:
        k["k{0}".format(num)]=i
        num = num + 1
    t1 = t2 = t3 = 0
    for i in range (1,12):
        t1=t1^int(k["k{0}".format(i)],2)
    for i in range (12,23):
        t2=t2^int(k["k{0}".format(i)],2)
    for i in range (23,33):
        t3=t3^int(k["k{0}".format(i)],2)   
    global x0 ,y0, z0
    x0=x0 + t1/256            
    y0=y0 + t2/256            
    z0=z0 + t3/256
    return x0,y0,z0