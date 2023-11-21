import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


root = tk.Tk()									
root.title("Image Processsing")
my_font = ('times', 18, 'bold')


#------------------------------------------------------------------------------------------------------
# Helper Functions
#Function 1  : uploading & Displaying input image
def upload ():
    f_types = [('JPG files','*.jpg'),('PNG files', '*.png'),('JPEG files', '*.jpeg')]
    filename = tk.filedialog.askopenfilename(filetypes = f_types)
    img = Image.open(filename) 
    img = img.resize((360,360))
    img1 = ImageTk.PhotoImage(img) #display it
    label = tk.Label(frame1)
    label.grid(row =1,column =1)
    label.image = img1
    label['image'] = img1
    return img


#Function 2  : Displaying output image
def display(img):
    img2 = Image.fromarray(img)
    img2 = ImageTk.PhotoImage(img2) #display it
    label = tk.Label(frame2)
    label.grid(row =1,column =1)
    label.image = img2
    label['image'] = img2

#Function 3 : Convert to gray scale
def RGB_gs(img):
   
    img_array = np.array(img)
    nimg_array=np.zeros(shape=(360,360))
    for i in range (len(img_array)):
        for j in range(len(img_array[i])):
            blue = img_array[i,j,0]
            green = img_array[i,j,1]
            red = img_array[i,j,2]
            grayscale_value = blue*0.114 + green*0.587 + red*0.299
            nimg_array[i,j] = grayscale_value
       
    return nimg_array;



#Function 4 : alpha value
def alpha(i):
    if (i == 0):
        return 1/np.sqrt(2)
    else:
        return 1

# Functions 5 & 6: Generate coefficients matrix 8*8 blocks
def DCT(img):

    #initializing the variables
    #----------------------------------
    n,m  =img.shape
    G = np.ones((n, m))
    #-----------------------------------

    #start the for loops of the DCT equation
    for u in range (0,8):
        for v in range (0,8):
            sum = 0      
            for x in range (0,8):
                for y in range (0,8):
                    #biulding the sum inside the DCT equation
                    sum = sum + np.cos(np.pi *u*(2*x+1)/16)  * np.cos(np.pi * v*(2*y+1)/16) * img[x,y]

            G[u,v] = (alpha(u)*alpha(v) * sum) / 4
                     

    return G


def IDCT (img):

    #initializing the variables
    #----------------------------------
    n,m  =img.shape
    f = np.ones((n, m))
    #-----------------------------------

    #start the for loops of the DCT equation
    for x in range (0,8):
        for y in range (0,8):
            sum = 0     
            for u in range (0,8):
                for v in range (0,8):
                       #biulding the sum inside the DCT equation
                     sum = sum + alpha(u) * alpha(v) * np.cos(np.pi *u*(2*x+1)/16)  * np.cos(np.pi * v*(2*y+1)/16) * img[u,v]

            f[x,y] = (np.round( (1/4) * sum )) +128
                     

    return f

#Function 8 :Quantize the matrix
def Quantize(input_arr):

    
    height, width = input_arr.shape[:2]
    quantized_img = np.zeros((height, width))
    #-----------------------------------------------------------

    quantize_mat = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                             [12, 12, 14, 19, 26, 58, 60, 55],
                             [14, 13, 16, 24, 40, 57, 69, 56],
                             [14, 17, 22, 29, 51, 87, 80, 62],
                             [18, 22, 37, 56, 68, 109, 103, 77],
                             [24, 35, 55, 64, 81, 104, 113, 92],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99]])

    #quantized_mat_inv = np.linalg.inv(quantize_mat)
    #quantized_img = int((input_arr @ quantized_mat_inv))
    n = quantize_mat.shape[0]
    quantized_img = np.zeros((n,n))
    for i in range (n):
        for j in range (n):
         quantized_img[i,j] = int(np.round(input_arr[i,j] / quantize_mat[i,j]))

    return quantized_img

# Function9 : Zigzag function for map the 8*8 array to 1*64 vector
# it group low frequency coefficients to the top level of the vector 
# and the high coefficient to the bottom --> to remove the large number of zeros
def zigzag(input_arr):

    #initializing the variables
    #----------------------------------
    h,v,i,vmin,hmin= 0,0,0,0,0
     
    vmax,hmax = input_arr.shape[:2]
    output_vec = np.zeros(( vmax * hmax))
    #----------------------------------

    while ((v < vmax) & (h < hmax)):

        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                output_vec[i] = input_arr[v, h]        # if we got to the first line
                if (h == hmax):
                    v +=  1
                else:
                    h += 1                        

                i += 1

            elif ((h == hmax -1 ) & (v < vmax)):   # if we got to the last column
                #print(2)
                output_vec[i] = input_arr[v, h] 
                v += 1
                i += 1

            elif ((v > vmin) & (h < hmax -1 )):    # all other cases
                #print(3)
                output_vec[i] = input_arr[v, h] 
                v -= 1
                h += 1
                i += 1
                 
        else:                                    # going down

            if ((v == vmax -1) & (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output_vec[i] = input_arr[v, h] 
                h += 1
                i += 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output_vec[i] = input_arr[v, h] 

                if (v == vmax -1):
                    h += 1
                else:
                    v += 1

                i += 1

            elif ((v < vmax -1) & (h > hmin)):     # all other cases
                #print(6)
                output_vec[i] = input_arr[v, h] 
                v += 1
                h -= 1
                i += 1

        if ((v == vmax-1) & (h == hmax-1)):          # bottom right element
            #print(7)        	
            output_vec[i] = input_arr[v, h] 
            break

    
    return output_vec


#Function10 : inverse zegzag method
def inverse_zigzag(input_vec, vmax, hmax):

    # initializing the variables
    #----------------------------------
    h,v,i,vmin,hmin= 0,0,0,0,0
    
    output_arr = np.zeros((vmax, hmax))
    #----------------------------------
    while ((v < vmax) and (h < hmax)): 

        if ((h + v) % 2) == 0:                 # going up     
            
            if (v == vmin):
                output_arr[v, h] = input_vec[i]        # if we got to the first line
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output_arr[v, h] = input_vec[i] 
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output_arr[v, h] = input_vec[i] 
                v = v - 1
                h = h + 1
                i = i + 1      
                
        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output_arr[v, h] = input_vec[i] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                output_arr[v, h] = input_vec[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output_arr[v, h] = input_vec[i] 
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output_arr[v, h] = input_vec[i] 
            break

    return output_arr




#Function 11 : flatten
def flatten(img): 
    img2= []
    for i in range (img.shape[0]):
        for j in range(img.shape[1]):
            img2.append(img[i,j])

    return np.array(img2)

#Function 12 : Reflatten:
def reflatten(img):
    img2 = []
    size = int(np.sqrt(img.shape[0]))
    for i in range (0,int(img.shape[0]),size):
        l=[]
        for j in range(size):
            l.append(img[i])
            i+=1
        img2.append(l)
    return np.array(img2)

#Function 13 : Distance:
def distance(x,centroid,k,img):
    
    d = np.zeros((k,x))
    
    for j in range(k):
        for i in range(x):
            d[j,i] = np.sqrt((img[i] - centroid[j])**2)
    return d

#Function 14 : Assigment
def Assigment(img,centroid,k):
    
    x = img.size
    
    d = distance(x,centroid,k,img)
    
    
    assigment = np.zeros(x)
    
    for i in range (x):
        assigment[i] = np.array(np.where(d[:,i] == np.amin(d[:,i])))[0,0]
        
    return assigment

#Function 15 : update centorid
def Upd_Cent(img,k,assigment,centroid):
    
    for j in range (k):
    
        #to get the index where belongs to the cluster k
        assigment2 = np.where(assigment==j)
    
        #for computing the mean to update centroids
        img2 = []
        for i in (assigment2):
            img2.append(img[i])
        
        centroid[j] = np.mean(img2)
    
    return centroid


#--------------------------------------------------------------------------------------
#Function  : Discrete Cosine Transform:
def DCT_algorithm():
    # upload image and display it 
    img = upload()

    # cnvert the input image into gray scale
    img = RGB_gs(img)

    # In this TP we devide the input image into 8-by-8 blocks
    # initializing the variables
    #----------------------------------

    block_size = 8
    height,width = img.shape[:2]         # shape 360x360
    block_y = height // 8                # 45 blocks on y axis
    block_x = width // 8                 # 45 blocks on x axis
    new_img = img.copy()
    #--------------------------------------
    
    # iterate over blocks
    for i in range(block_y):
        for j in range(block_x):

            # START ENCODING:

            # select the current block we want to process using calculated indices
            block = new_img[ 8*i : 8*(i+1) , 8*j : 8*(j+1) ]
            
            # apply 2D discrete cosine transform to the selected block
            DCT_img = DCT(block)
            
            # Quantize array
            qua_img = Quantize(DCT_img)

            # reorder DCT coefficients in zig zag order 
            # note : it will give you a one dimentional array (here: 64)
            reordered1 = zigzag(qua_img)

            # START DECODING:
            # use inverse_zigzag function to scan and reorder the array into a block
            reordered2 = inverse_zigzag(reordered1, int(block_size), int(block_size))

            # apply 2D inverse discrete cosine transform to the reordered matrix
            IDCT_arr = IDCT(reordered2)
            
            
            # copy reshaped matrix into padded_img on current block corresponding indices
            new_img[8*i : 8*(i+1) , 8*j : 8*(j+1)] = IDCT_arr


    #print('quantized array: ',qua_img)
    # Display the reconstruct image
    display(new_img)



#Function : K-means
def K_means():
    img = upload()
    img_array = RGB_gs(img)

    k = int(retrieve_input(textBox))
    
    #convert the image into one vector
    img2 = flatten(img_array)

    #initialise centroid randomly
    centroid = np.random.randint(0,255,k)
    #print('\nthe centroids are :\n',centroid)
    
    #get the 1st assigment
    assigment1 = Assigment(img2,centroid,k)
    #print("\nThe assigment is :\n",assigment1)
    
    #update the centroid, comput the 2nd ones
    centroid = Upd_Cent(img2,k,assigment1,centroid)
    #print('\nthe updated centoids are:\n',centroid)


    i=0
    while (i<1000):
         
        #computing the new assigment
        assigment2 = Assigment(img2,centroid,k)
    
        #comparing the previous assigment with the new one
        #if they are not the same:
        if ((assigment2 != assigment1).all()):
            
            #update the centroids
            centroid = Upd_Cent(img2,k,assigment2,centroid)
            
            #delete the before-last assigment and save the last one
            assigment1 = assigment2.copy()
            
            i += 1     
        else:
            break
    

    #showing the cluster 0
    cluster = int(retrieve_input(textBox2))
    
    #getting the index of the pixels which are in cluster c (that we want to show)
    index = (np.transpose(np.array(np.where(assigment1 == cluster))[0].tolist()))
 
    #the pixels from the same cluster that we want to show are in white and the others are in black
    img3 = img2.copy()
    j = 0
    for i in range (img2.size):
        if (j< index.size):
            if ((i == index[j]).all()):
                img3[i] = 255
                j+=1
            else:
                img3[i] = 0
        else:
            img3[i] = 0
    
    #reflatten img3 to show the cluster 
    img4 = reflatten(img3)
    display(img4)
    #return the 
#---------------------------------------------------------------------------------------------------------------

canvas = tk.Canvas(root,height = 600, width = 1200, bg = '#D8BFD8')
canvas.pack()

frame1 = tk.Frame(root, bg ="white")
frame1.place(relwidth =0.3, relheight = 0.6,relx = 0.1, rely = 0.18  )


frame2 = tk.Frame(root, bg ="white")
frame2.place(relwidth =0.3, relheight = 0.6,relx = 0.6, rely = 0.18 )

Upl = Label (root,text = 'uploaded image',bg = '#D8BFD8' )
Upl.config(font=('Helvatical',13))
Upl.place(x=230,y=475) 

Out = Label (root,text = 'Output image',bg = '#D8BFD8' )
Out.config(font=('Helvatical',13))
Out.place(x=850,y=475) 

Title = Label(root,text='TP Digital Image ',bg = '#D8BFD8')
Title.config(font=('Helvatical',30))
Title.place(x =470, y= 30)

Note = Label(root,text ='Note:',bg = '#D8BFD8')
Note.place(x =90, y= 550)

Ins = Label(root,text = 'click on button so you can play the function.',bg = '#D8BFD8')
Ins.place(x =75, y= 570)


########################################################################################################################


radio = IntVar()

LGauss = tk.Button(root, text = "DCT_algorithm" , padx=48, pady=5, command=lambda :  DCT_algorithm())
LGauss.place(x =510, y= 200)


def retrieve_input(textBox):
    inputValue=textBox.get("1.0",END)
    return inputValue


k_number = Label(root,text = 'Enter number of clusters.',bg = '#D8BFD8')
k_number.place(x =520, y= 240)
textBox=Text(root, height=1, width=5)
textBox.place(x =580, y= 260)

c_cluster = Label(root,text = 'Enter the cluster you want to show.',bg = '#D8BFD8')
c_cluster.place(x =520, y= 280)
textBox2=Text(root, height=1, width=5)
textBox2.place(x =580, y= 300)

Canny = tk.Button(root, text = "K means clustering" , padx=38, pady=5, command=lambda: K_means())
Canny.place(x =510, y= 325)

root.mainloop()




        