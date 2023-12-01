# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:15:27 2020

@author: Andrew
"""

import cv2
from PIL import Image
import os
from os import listdir
import os.path
from os.path import isfile, join
import numpy
from etree import ElementTree as ET
from ElementTree_pretty import prettify
import numpy as np
import glob

def annotating_images(images_path,test_percentage,image_width,image_height,number_classes,label_list,dish_name,adding_to_dataset,folder_name_images,resize_image_width,resize_image_height,type_image):
    # Delete Images, in folders
    if adding_to_dataset==1:
        print('Adding to Dataset')
    else:
        folder_names=[glob.glob(images_path+'/only_images/*'),
                      glob.glob(images_path+'/only_images_blur/*'),
                      glob.glob(images_path+'/only_images_90/*'),
                      glob.glob(images_path+'/only_images_180/*'),
                      glob.glob(images_path+'/only_images_270/*'),
                      glob.glob(images_path+'/xml/*'),
                      glob.glob(images_path+'/xml_blur/*'),
                      glob.glob(images_path+'/xml_90/*'),
                      glob.glob(images_path+'/xml_180/*'),
                      glob.glob(images_path+'/xml_270/*'),
                      glob.glob(images_path+'/xml_all/*.jpg'),
                      glob.glob(images_path+'/xml_all/*.xml'),
                      glob.glob(images_path+'/xml_all/train/*'),
                      glob.glob(images_path+'/xml_all/test/*')]
    
        for folder in folder_names:
            for f_ in folder:
                os.remove(f_)

    path=images_path+'/'+folder_name_images
    files = [ i for i in listdir(path) if isfile(join(path,i)) ]
    # Saving images to all eleven folders
    for fov in range(len(files)):
        new=cv2.imread(join(path,files[fov]))
        cv2.imwrite(join(images_path+'/only_images',files[fov]),new)
        cv2.imwrite(join(images_path+'/only_images_blur',files[fov]),new)
        cv2.imwrite(join(images_path+'/only_images_90',files[fov]),new)
        cv2.imwrite(join(images_path+'/only_images_180',files[fov]),new)
        cv2.imwrite(join(images_path+'/only_images_270',files[fov]),new)
        cv2.imwrite(join(images_path+'/xml',files[fov]),new)
        cv2.imwrite(join(images_path+'/xml_blur',files[fov]),new)
        cv2.imwrite(join(images_path+'/Petri_xml_90',files[fov]),new)
        cv2.imwrite(join(images_path+'/Petri_xml_180',files[fov]),new)
        cv2.imwrite(join(images_path+'/Petri_xml_270',files[fov]),new)
        cv2.imwrite(join(images_path+'/Petri_xml_all',files[fov]),new)
    
    print('Done saving images from dishes')
    
    path_only=images_path+'/only_images'
    xml=images_path+'/xml'
    files_only = [ i for i in listdir(path_only) if isfile(join(path_only,i)) ]
    imgp = numpy.empty(len(files_only), dtype=object)
    imp = numpy.empty(len(files_only), dtype=object)
    new = numpy.empty(len(files), dtype=object)
    width_p = numpy.empty(len(files_only), dtype=object)
    height_p = numpy.empty(len(files_only), dtype=object)
    
    path_blur=images_path+'/only_images_blur'
    xml_blur=images_path+'/xml_blur'
    files_blur = [ i for i in listdir(path_blur) if isfile(join(path_blur,i)) ]
    imgp_blur = numpy.empty(len(files_blur), dtype=object)
    imp_blur = numpy.empty(len(files_blur), dtype=object)
    new_blur = numpy.empty(len(files_blur), dtype=object)
    width_p_blur = numpy.empty(len(files_blur), dtype=object)
    height_p_blur = numpy.empty(len(files_blur), dtype=object)
    
    path_90=images_path+'/only_images_90'
    xml_90=images_path+'/xml_90'
    files_90 = [ i for i in listdir(path_90) if isfile(join(path_90,i)) ]
    imgp_90 = numpy.empty(len(files_90), dtype=object)
    imp_90 = numpy.empty(len(files_90), dtype=object)
    new_90 = numpy.empty(len(files_90), dtype=object)
    width_p_90 = numpy.empty(len(files_90), dtype=object)
    height_p_90 = numpy.empty(len(files_90), dtype=object)
    
    path_180=images_path+'/only_images_180'
    xml_180=images_path+'/xml_180'
    files_180 = [ i for i in listdir(path_180) if isfile(join(path_180,i)) ]
    imgp_180 = numpy.empty(len(files_180), dtype=object)
    imp_180 = numpy.empty(len(files_180), dtype=object)
    new_180 = numpy.empty(len(files_180), dtype=object)
    width_p_180 = numpy.empty(len(files_180), dtype=object)
    height_p_180 = numpy.empty(len(files_180), dtype=object)
    
    path_270=images_path+'/only_images_270'
    xml_270=images_path+'/xml_270'
    files_270 = [ i for i in listdir(path_270) if isfile(join(path_270,i)) ]
    imgp_270 = numpy.empty(len(files_270), dtype=object)
    imp_270 = numpy.empty(len(files_270), dtype=object)
    new_270 = numpy.empty(len(files_270), dtype=object)
    width_p_270 = numpy.empty(len(files_270), dtype=object)
    height_p_270 = numpy.empty(len(files_270), dtype=object)
    
    xml_all=images_path+'/xml_all'
    
    x_s = [ [] for z in range(len(files)) ]
    y_s = [ [] for z in range(len(files)) ] 
    w   = [ [] for z in range(len(files)) ]
    h   = [ [] for z in range(len(files)) ]
    x_s2 = [ [] for z in range(len(files)) ]
    y_s2 = [ [] for z in range(len(files)) ]
    
    labels = [ [] for z in range(len(files)) ]
    labels_blur = [ [] for z in range(len(files_blur)) ]
    labels_90 = [ [] for z in range(len(files_90)) ]
    labels_180 = [ [] for z in range(len(files_180)) ]
    labels_270 = [ [] for z in range(len(files_270)) ]
    
    f = numpy.empty(len(files), dtype=object)
    k = numpy.empty(len(files_blur), dtype=object)
    v = numpy.empty(len(files_90), dtype=object)
    g = numpy.empty(len(files_180), dtype=object)
    b = numpy.empty(len(files_270), dtype=object)
    
    f_all = numpy.empty(len(files), dtype=object)
    k_all = numpy.empty(len(files_blur), dtype=object)
    v_all = numpy.empty(len(files_90), dtype=object)
    g_all = numpy.empty(len(files_180), dtype=object)
    b_all = numpy.empty(len(files_270), dtype=object)
        
    center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
    scale=1
    fromCenter=False
    for p in range(len(files)):
        print('Image Number = ',p+1)
        # Original Image
        imgp[p] = cv2.imread( join(path,files[p]) )
        imp[p] = Image.open(join(path,files[p]))
        width_p[p], height_p[p] = imp[p].size
        new[p]=imgp[p]
        new[p]=cv2.resize(imgp[p],(resize_image_width,resize_image_height))
        cv2.imwrite(xml+'/'+dish_name+files[p],new[p])
        cv2.imwrite(xml_all+'/'+dish_name+files[p],new[p])
        imp[p].close()
        
        # Blur Image
        imgp_blur[p] = cv2.imread( join(path_blur,files_blur[p]) )
        imp_blur[p] = Image.open(join(path_blur,files_blur[p]))
        width_p_blur[p], height_p_blur[p] = imp_blur[p].size
        new_blur[p]=cv2.GaussianBlur(imgp_blur[p],(5,5),0)
        cv2.imwrite(xml_blur+'/'+dish_name+'blur_'+files_blur[p],new_blur[p])
        cv2.imwrite(xml_all+'/'+dish_name+'blur_'+files_blur[p],new_blur[p])
        imp_blur[p].close
        
        # Rotate Image by 90 degrees
        imgp_90[p] = cv2.imread( join(path_90,files_90[p]) )
        imp_90[p] = Image.open(join(path_90,files_90[p]))
        width_p_90[p], height_p_90[p] = imp_90[p].size
        M_90 = cv2.getRotationMatrix2D(center,90, scale)
        cosine = np.abs(M_90[0, 0])
        sine = np.abs(M_90[0, 1])
        nW = int((image_height * sine) + (image_height * cosine))
        nH = int((image_height * cosine) + (image_width * sine))
        M_90[0, 2] += (nW / 2) - int((float(image_width))/(2))
        M_90[1, 2] += (nH / 2) - int((float(image_height))/(2))
        new_90[p]=cv2.warpAffine(imgp[p], M_90, (image_height, image_width)) 
        cv2.imwrite(xml_90+'/'+dish_name+'90_degree_'+files_90[p],new_90[p])
        cv2.imwrite(xml_all+'/'+dish_name+'90_degree_'+files_90[p],new_90[p])
        imp_90[p].close()
        
        # Rotate Image by 180 degrees
        imgp_180[p] = cv2.imread( join(path_180,files_180[p]) )
        imp_180[p] = Image.open(join(path_180,files_180[p]))
        width_p_180[p], height_p_180[p] = imp_180[p].size
        M_180 = cv2.getRotationMatrix2D(center,180, scale)
        cosine = np.abs(M_180[0, 0])
        sine = np.abs(M_180[0, 1])
        nW = int((image_height * sine) + (image_height * cosine))
        nH = int((image_height * cosine) + (image_width * sine))
        M_180[0, 2] += (nW / 2) - int((float(image_width))/(2))
        M_180[1, 2] += (nH / 2) - int((float(image_height))/(2))
        new_180[p]=cv2.warpAffine(imgp[p], M_180, (image_height, image_width))
        cv2.imwrite(xml_180+'/'+dish_name+'180_degree_'+files_180[p],new_180[p])
        cv2.imwrite(xml_all+'/'+dish_name+'180_degree_'+files_180[p],new_180[p])
        imp_180[p].close()
        
        # Rotate Image by 270 degrees
        imgp_270[p] = cv2.imread( join(path_270,files_270[p]) )
        imp_270[p] = Image.open(join(path_270,files_270[p]))
        width_p_270[p], height_p_270[p] = imp_270[p].size
        M_270 = cv2.getRotationMatrix2D(center,270, scale)
        cosine = np.abs(M_270[0, 0])
        sine = np.abs(M_270[0, 1])
        nW = int((image_height * sine) + (image_height * cosine))
        nH = int((image_height * cosine) + (image_width * sine))
        M_270[0, 2] += (nW / 2) - int((float(image_width))/(2))
        M_270[1, 2] += (nH / 2) - int((float(image_height))/(2))
        new_270[p]=cv2.warpAffine(imgp[p], M_270, (image_height, image_width)) 
        cv2.imwrite(xml_270+'/'+dish_name+'270_degree_'+files_270[p],new_270[p])
        cv2.imwrite(xml_all+'/'+dish_name+'270_degree_'+files_270[p],new_270[p])
        imp_270[p].close()
        
        cv2.imshow('Image',new[p])
        cv2.waitKey(10)
        if number_classes==1:
            image=input('How many {}? '.format(label_list[0]))
            image=int(float(image))
        else:
            #uncomment for whole dish
            if type_image=='embryo':
                image=3
            else:
                image=input('How many '+str(label_list[0:len(label_list)])[1:-1]+'? ')
                image=int(float(image))
    
        if image==0:
            os.remove(join(xml,files_only[p]))
            os.remove(join(xml_blur,dish_name+'blur_'+files_blur[p]))
            os.remove(join(xml_90,dish_name+'90_degree_'+files_90[p]))
            os.remove(join(xml_180,dish_name+'180_degree_'+files_180[p]))
            os.remove(join(xml_270,dish_name+'270_degree_'+files_270[p]))
            os.remove(join(xml_all,dish_name+files_only[p]))
            os.remove(join(xml_all,dish_name+'blur_'+files_blur[p]))
            os.remove(join(xml_all,dish_name+'90_degree_'+files_90[p]))
            os.remove(join(xml_all,dish_name+'180_degree_'+files_180[p]))
            os.remove(join(xml_all,dish_name+'270_degree_'+files_270[p]))
            continue
        else:
            imgp[p] = cv2.imread( join(path,files[p]))
            imp[p] = Image.open(join(path,files[p]))
            width_p[p], height_p[p] = imp[p].size
            imp[p].close()
            new[p]=cv2.resize(imgp[p],(resize_image_width,resize_image_height))
            for ROIS in range(0,image):
                R=0
                D=0
                V=0
                G=0
                B=0
                # cropping image
                (x_ss,y_ss,ws,hs) = cv2.selectROI("Image",new[p],fromCenter)
                cv2.rectangle(new[p],(x_ss,y_ss),(x_ss+ws,y_ss+hs),(0,255,0),1)
                if number_classes==1:
                    label=label_list[0]
                else:
                    image_ne=input(str(label_list[0:len(label_list)])[1:-1]+'? ')
                    image_ne=int(float(image_ne))
                    label=label_list[image_ne-1]

                x_ss2=(x_ss+ws)
                y_ss2=(y_ss+hs)
                x_s[p].append(int(x_ss*(image_width/resize_image_width)))
                y_s[p].append(int(y_ss*(image_height/resize_image_height)))
                w[p].append(ws)
                h[p].append(hs)
                x_s2[p].append(int(x_ss2*(image_width/resize_image_width)))
                y_s2[p].append(int(y_ss2*(image_height/resize_image_height)))
                
                # Writing xml file
                labels[p].append(label)            
                top = ET.Element("annotations")
                folder=ET.SubElement(top,'folder')
                folder.text='xml'        
                filename=ET.SubElement(top,'filename')
                filename.text= join(dish_name+files_only[p])
                path_et=ET.SubElement(top,'path')
                path_et.text=join(xml_all,dish_name+files_only[p])
                size=ET.SubElement(top,'size')
                width=ET.SubElement(size,'width')
                height=ET.SubElement(size,'height')
                width.text=str(width_p[p])
                height.text=str(height_p[p])
                for coord in x_s[p]:
                    objectt = ET.SubElement(top,"object")
                    name = ET.SubElement(objectt,"name")
                    name.text=labels[p][R]
                    bndbox = ET.SubElement(objectt,"bndbox")
                    xmin = ET.SubElement(bndbox,"xmin")
                    xmax = ET.SubElement(bndbox,"xmax")
                    ymin = ET.SubElement(bndbox,"ymin")
                    ymax = ET.SubElement(bndbox,"ymax")
                    xmin.text = str(x_s[p][R])
                    xmax.text = str(x_s2[p][R])
                    ymin.text = str(y_s[p][R])
                    ymax.text = str(y_s2[p][R])
                    R+=1
    
                labels_blur[p].append(label)            
                top_blur = ET.Element("annotations")
                folder_blur=ET.SubElement(top_blur,'folder')
                folder_blur.text='xml_blur'        
                filename_blur=ET.SubElement(top_blur,'filename')
                filename_blur.text= join(dish_name+'blur_'+files_blur[p])
                path_blur_et=ET.SubElement(top_blur,'path')
                path_blur_et.text=join(xml_all,dish_name+'blur_'+files_blur[p])
                size_blur=ET.SubElement(top_blur,'size')
                width_blur=ET.SubElement(size_blur,'width')
                height_blur=ET.SubElement(size_blur,'height')
                width_blur.text=str(image_width)
                height_blur.text=str(image_height)
                for coord in x_s[p]:
                    objectt_blur = ET.SubElement(top_blur,"object")
                    name_blur = ET.SubElement(objectt_blur,"name")
                    name_blur.text=labels_blur[p][D]
                    bndbox_blur = ET.SubElement(objectt_blur,"bndbox")
                    xmin = ET.SubElement(bndbox_blur,"xmin")
                    xmax = ET.SubElement(bndbox_blur,"xmax")
                    ymin = ET.SubElement(bndbox_blur,"ymin")
                    ymax = ET.SubElement(bndbox_blur,"ymax")
                    xmin.text = str(x_s[p][D])
                    xmax.text = str(x_s2[p][D])
                    ymin.text = str(y_s[p][D])
                    ymax.text = str(y_s2[p][D])
                    D+=1
    
                labels_90[p].append(label)            
                top_90 = ET.Element("annotations")
                folder_90=ET.SubElement(top_90,'folder')
                folder_90.text='xml_90'        
                filename_90=ET.SubElement(top_90,'filename')
                filename_90.text= join(dish_name+'90_degree_'+files_90[p])
                path_90_et=ET.SubElement(top_90,'path')
                path_90_et.text=join(xml_all,dish_name+'90_degree_'+files_90[p])
                size_90=ET.SubElement(top_90,'size')
                width_90=ET.SubElement(size_90,'width')
                height_90=ET.SubElement(size_90,'height')
                width_90.text=str(image_width)
                height_90.text=str(image_height)
                for coord in x_s[p]:
                    objectt_90 = ET.SubElement(top_90,"object")
                    name_90 = ET.SubElement(objectt_90,"name")
                    name_90.text=labels_90[p][V]
                    bndbox_90 = ET.SubElement(objectt_90,"bndbox")
                    xmin90 = ET.SubElement(bndbox_90,"xmin")
                    xmax90 = ET.SubElement(bndbox_90,"xmax")
                    ymin90 = ET.SubElement(bndbox_90,"ymin")
                    ymax90 = ET.SubElement(bndbox_90,"ymax")
                    x_mini_90,y_maxi_90=np.matrix(M_90)*np.array([[x_s[p][V]],[y_s[p][V]],[1]])
                    x_maxi_90,y_mini_90=np.matrix(M_90)*np.array([[x_s2[p][V]],[y_s2[p][V]],[1]])
                    xmin90.text = str(int(x_mini_90[0,0]))
                    xmax90.text = str(int(x_maxi_90[0,0]))
                    ymin90.text = str(int(y_mini_90[0,0]))
                    ymax90.text = str(int(y_maxi_90[0,0]))
                    V+=1
                    
                labels_180[p].append(label)            
                top_180 = ET.Element("annotations")
                folder_180=ET.SubElement(top_180,'folder')
                folder_180.text='xml_180'        
                filename_180=ET.SubElement(top_180,'filename')
                filename_180.text= join(dish_name+'180_degree_'+files_180[p])
                path_180_et=ET.SubElement(top_180,'path')
                path_180_et.text=join(xml_all,dish_name+'180_degree_'+files_180[p])
                size_180=ET.SubElement(top_180,'size')
                width_180=ET.SubElement(size_180,'width')
                height_180=ET.SubElement(size_180,'height')
                width_180.text=str(image_width)
                height_180.text=str(image_height)
                for coord in x_s[p]:
                    objectt_180 = ET.SubElement(top_180,"object")
                    name_180 = ET.SubElement(objectt_180,"name")
                    name_180.text=labels_180[p][G]
                    bndbox_180 = ET.SubElement(objectt_180,"bndbox")
                    xmin180 = ET.SubElement(bndbox_180,"xmin")
                    xmax180 = ET.SubElement(bndbox_180,"xmax")
                    ymin180 = ET.SubElement(bndbox_180,"ymin")
                    ymax180 = ET.SubElement(bndbox_180,"ymax")
                    x_maxi_180,y_maxi_180=np.matrix(M_180)*np.array([[x_s[p][G]],[y_s[p][G]],[1]])
                    x_mini_180,y_mini_180=np.matrix(M_180)*np.array([[x_s2[p][G]],[y_s2[p][G]],[1]])
                    xmin180.text = str(int(x_mini_180[0,0]))
                    xmax180.text = str(int(x_maxi_180[0,0]))
                    ymin180.text = str(int(y_mini_180[0,0]))
                    ymax180.text = str(int(y_maxi_180[0,0]))
                    G+=1
                    
                labels_270[p].append(label)            
                top_270 = ET.Element("annotations")
                folder_270=ET.SubElement(top_270,'folder')
                folder_270.text='xml_270'        
                filename_270=ET.SubElement(top_270,'filename')
                filename_270.text= join(dish_name+'270_degree_'+files_270[p])
                path_270_et=ET.SubElement(top_270,'path')
                path_270_et.text=join(xml_all,dish_name+'270_degree_'+files_270[p])
                size_270=ET.SubElement(top_270,'size')
                width_270=ET.SubElement(size_270,'width')
                height_270=ET.SubElement(size_270,'height')
                width_270.text=str(image_width)
                height_270.text=str(image_height)
                for coord in x_s[p]:
                    objectt_270 = ET.SubElement(top_270,"object")
                    name_270 = ET.SubElement(objectt_270,"name")
                    name_270.text=labels_270[p][B]
                    bndbox_270 = ET.SubElement(objectt_270,"bndbox")
                    xmin270 = ET.SubElement(bndbox_270,"xmin")
                    xmax270 = ET.SubElement(bndbox_270,"xmax")
                    ymin270 = ET.SubElement(bndbox_270,"ymin")
                    ymax270 = ET.SubElement(bndbox_270,"ymax")
                    x_maxi_270,y_mini_270=np.matrix(M_270)*np.array([[x_s[p][B]],[y_s[p][B]],[1]])
                    x_mini_270,y_maxi_270=np.matrix(M_270)*np.array([[x_s2[p][B]],[y_s2[p][B]],[1]])
                    xmin270.text = str(int(x_mini_270[0,0]))
                    xmax270.text = str(int(x_maxi_270[0,0]))
                    ymin270.text = str(int(y_mini_270[0,0]))
                    ymax270.text = str(int(y_maxi_270[0,0]))
                    B+=1
                    
                completeName_1 = os.path.join(xml,dish_name + "myxmlfile_{}.xml".format(p+1))
                f[p] =  open(completeName_1, "w")
                f[p].write(prettify(top))
                f[p].close()
                completeName_2 = os.path.join(xml_blur, dish_name +"myxmlfile_blur_{}.xml".format(p+1))
                k[p] =  open(completeName_2, "w")
                k[p].write(prettify(top_blur))
                k[p].close()
                completeName_3 = os.path.join(xml_90,dish_name + "myxmlfile_90_{}.xml".format(p+1))
                v[p] =  open(completeName_3, "w")
                v[p].write(prettify(top_90))
                v[p].close()
                completeName_4 = os.path.join(xml_180,dish_name + "myxmlfile_180_{}.xml".format(p+1))
                g[p] =  open(completeName_4, "w")
                g[p].write(prettify(top_180))
                g[p].close()
                completeName_5 = os.path.join(xml_270,dish_name + "myxmlfile_270_{}.xml".format(p+1))
                b[p] =  open(completeName_5, "w")
                b[p].write(prettify(top_270))
                b[p].close()
                completeName_6 = os.path.join(xml_all, dish_name +"myxmlfile_{}.xml".format(p+1))
                f_all[p] =  open(completeName_6, "w")
                f_all[p].write(prettify(top))
                f_all[p].close()
                completeName_7 = os.path.join(xml_all,dish_name + "myxmlfile_blur_{}.xml".format(p+1))
                k_all[p] =  open(completeName_7, "w")
                k_all[p].write(prettify(top_blur))
                k_all[p].close()
                completeName_8 = os.path.join(xml_all,dish_name + "myxmlfile_90_{}.xml".format(p+1))
                v_all[p] =  open(completeName_8, "w")
                v_all[p].write(prettify(top_90))
                v_all[p].close()
                completeName_9 = os.path.join(xml_all, dish_name +"myxmlfile_180_{}.xml".format(p+1))
                g_all[p] =  open(completeName_9, "w")
                g_all[p].write(prettify(top_180))
                g_all[p].close()
                completeName_10 = os.path.join(xml_all, dish_name +"myxmlfile_270_{}.xml".format(p+1))
                b_all[p] =  open(completeName_10, "w")
                b_all[p].write(prettify(top_270))
                b_all[p].close()
                
                # saving xml file
                if p<int(len(files)*(float(test_percentage)/float(100))):
                    cv2.imwrite(xml_all+'/'+'test/'+dish_name+files_only[p],imgp[p])
                    cv2.imwrite(xml_all+'/'+'test/'+dish_name+'blur_'+files_blur[p],new_blur[p])
                    cv2.imwrite(xml_all+'/'+'test/'+dish_name+'90_degree_'+files_90[p],new_90[p])
                    cv2.imwrite(xml_all+'/'+'test/'+dish_name+'180_degree_'+files_180[p],new_180[p])
                    cv2.imwrite(xml_all+'/'+'test/'+dish_name+'270_degree_'+files_270[p],new_270[p])
    
                    completeName_6 = os.path.join(xml_all+'/'+'test',dish_name + "myxmlfile_{}.xml".format(p+1))
                    f_all[p] =  open(completeName_6, "w")
                    f_all[p].write(prettify(top))
                    f_all[p].close()
                    completeName_7 = os.path.join(xml_all+'/'+'test', dish_name +"myxmlfile_blur_{}.xml".format(p+1))
                    k_all[p] =  open(completeName_7, "w")
                    k_all[p].write(prettify(top_blur))
                    k_all[p].close()
                    completeName_8 = os.path.join(xml_all+'/'+'test', dish_name +"myxmlfile_90_{}.xml".format(p+1))
                    v_all[p] =  open(completeName_8, "w")
                    v_all[p].write(prettify(top_90))
                    v_all[p].close()
                    completeName_9 = os.path.join(xml_all+'/'+'test', dish_name +"myxmlfile_180_{}.xml".format(p+1))
                    g_all[p] =  open(completeName_9, "w")
                    g_all[p].write(prettify(top_180))
                    g_all[p].close()
                    completeName_10 = os.path.join(xml_all+'/'+'test',dish_name + "myxmlfile_270_{}.xml".format(p+1))
                    b_all[p] =  open(completeName_10, "w")
                    b_all[p].write(prettify(top_270))
                    b_all[p].close()
                else:
                    cv2.imwrite(xml_all+'/'+'train/'+dish_name+files_only[p],imgp[p])
                    cv2.imwrite(xml_all+'/'+'train/'+dish_name+'blur_'+files_blur[p],new_blur[p])
                    cv2.imwrite(xml_all+'/'+'train/'+dish_name+'90_degree_'+files_90[p],new_90[p])
                    cv2.imwrite(xml_all+'/'+'train/'+dish_name+'180_degree_'+files_180[p],new_180[p])
                    cv2.imwrite(xml_all+'/'+'train/'+dish_name+'270_degree_'+files_270[p],new_270[p])
                    
                    completeName_6 = os.path.join(xml_all+'/'+'train', dish_name +"myxmlfile_{}.xml".format(p+1))
                    f_all[p] =  open(completeName_6, "w")
                    f_all[p].write(prettify(top))
                    f_all[p].close()
                    completeName_7 = os.path.join(xml_all+'/'+'train', dish_name +"myxmlfile_blur_{}.xml".format(p+1))
                    k_all[p] =  open(completeName_7, "w")
                    k_all[p].write(prettify(top_blur))
                    k_all[p].close()
                    completeName_8 = os.path.join(xml_all+'/'+'train', dish_name +"myxmlfile_90_{}.xml".format(p+1))
                    v_all[p] =  open(completeName_8, "w")
                    v_all[p].write(prettify(top_90))
                    v_all[p].close()
                    completeName_9 = os.path.join(xml_all+'/'+'train', dish_name +"myxmlfile_180_{}.xml".format(p+1))
                    g_all[p] =  open(completeName_9, "w")
                    g_all[p].write(prettify(top_180))
                    g_all[p].close()
                    completeName_10 = os.path.join(xml_all+'/'+'train', dish_name +"myxmlfile_270_{}.xml".format(p+1))
                    b_all[p] =  open(completeName_10, "w")
                    b_all[p].write(prettify(top_270))
                    b_all[p].close()

# # Uncomment this to run for petri dishes
# images_path='C:/Users/User/Downloads/common_carp_petri_dishes'
# test_percentage=25
# type_image='petri_dish'
# image_width=4608
# image_height=3072
# resize_image_width=1000
# resize_image_height=667
# number_classes=2
# label_list=['embryo','bubble']
# dish_name='Sample_1'
# adding_to_dataset=0
# folder_name_images='All_images'
# annotating_images(images_path,test_percentage,image_width,image_height,number_classes,label_list,dish_name,adding_to_dataset,folder_name_images,resize_image_width,resize_image_height,type_image)

# # # Uncomment this to run for just embryos
# # images_path='C:/Users/Andrew/anaconda3/Example_of_Annotations_Petri_Dish'
# # test_percentage=25
# # type_image='embryo'
# # image_width=350
# # image_height=400
# # resize_image_width=350
# # resize_image_height=400
# # number_classes=3
# # label_list=['anterior','centroid','posterior']
# # dish_name='dish_1'
# # adding_to_dataset=0
# # folder_name_images='All_images'
# # annotating_images(images_path,test_percentage,image_width,image_height,number_classes,label_list,dish_name,adding_to_dataset,folder_name_images,resize_image_width,resize_image_height,type_image)

