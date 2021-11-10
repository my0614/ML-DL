
import os
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import albumentations as A
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32, GDT_Byte, GDT_UInt16
import class_id
from osgeo import gdal
from osgeo import ogr# .shp 파일을 raster .tif으로 변환 + 크롭



def writeTiff(outpath, imgbuf, proj, gt=None):
        h = imgbuf.shape[0]
        w = imgbuf.shape[1]

        bcnt = 1
        if len(imgbuf.shape) > 2:
            bcnt = imgbuf.shape[2]

        # save tiff
        driver = gdal.GetDriverByName("GTiff")
        
        gtype = -1
        if imgbuf.dtype == np.uint8:
            gtype = GDT_Byte
        elif imgbuf.dtype == np.uint16:
            gtype = GDT_UInt16
        elif imgbuf.dtype == np.float32:
            gtype = GDT_Float32
        else: 
            sys.exit('tif image to save must either be uint8, uint16, or float32')

        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outDataset = driver.Create(outpath, w, h, bcnt, gtype)

        # # if gt is not None:
        # outDataset.SetGeoTransform(gt)
        # outDataset.SetProjection(proj)

        for b in range(bcnt):
            if len(imgbuf.shape) > 2:
                outBand = outDataset.GetRasterBand(b+1)
                outBand.WriteArray(imgbuf[:,:,b], 0, 0)
            else:
                outBand = outDataset.GetRasterBand(1)
                outBand.WriteArray(imgbuf, 0, 0)
        outBand.FlushCache()

def _stretch(im, p1, p2): 
    # print('p1, p2: ', p1, p2)
    # if p1 == p2: 
    #     print(f'p1, p2 are the same ({p1, p2})')
    #     print(np.unique(im), im.shape)
    im = im.astype(np.float32)
    im = (im - p1) / float(p2 - p1)
    im = np.clip(im, 0, 1.0)

    return im


class Augmentation:
    @staticmethod
    def testing(file_names,img_dir,transform):
        cnt_list = {}
        cnt = 0
        for file_name in file_names:
            batch_x = []
            batch_y = []
            #file_name    = self.x[i]
            GT_file_name = file_name
            
            #GT_file_name = file_name[0:6]+'y'+file_name[7:]


            src_dataset = gdal.Open(os.path.join(img_dir['im_dir'],file_name),gdal.GA_ReadOnly)
            GT_dataset  = gdal.Open(os.path.join(img_dir['label_dir'],GT_file_name),gdal.GA_ReadOnly)

            src_img = src_dataset.ReadAsArray()
            GT_img = GT_dataset.ReadAsArray()
            
            #print(GT_img.astype(np.uint8))
            #print(GT_img.shape)

            for i in range(512):
                for j in range(512):
                    if int(GT_img[i,j]) == 5:
                        cnt += 1

            cnt_list[GT_file_name] = cnt
            cnt = 0

        cnt_list = sorted(cnt_list.items(), key=lambda x: x[1], reverse=True) # value값으로 정렬
        
        end = int(len(cnt_list)/10)
        
        pick_list = cnt_list[: end +1]
        #pick_list = cnt_list[:int(len(cnt_list)*0.1), 0]
        x_data_path = ''
        y_data_path = ''

        for file_name in pick_list:
            # print(file_name[0])
            batch_x = []
            batch_y = []
            #file_name    = self.x[i]
            GT_file_name = file_name
            
            #GT_file_name = file_name[0:6]+'y'+file_name[7:]

            print('file_name',os.path.join(x_data_path,file_name[0]))
            src_dataset = gdal.Open(os.path.join(x_data_path,file_name[0]),gdal.GA_ReadOnly)
            GT_dataset  = gdal.Open(os.path.join(y_data_path,GT_file_name[0]),gdal.GA_ReadOnly)

            src_img = src_dataset.ReadAsArray()
            GT_img = GT_dataset.ReadAsArray()
            
            
            print('src_img',src_img.shape)
            print('GT_img',GT_img.shape)
            src_img = np.transpose(src_img,(1,2,0)) # transe shape
            print('src_img',src_img.shape)
            
            #print(cnt_list)     
            #BGR -> RGB
            # ----- src_img = np.transpose(src_img, (1, 2, 0))
            # plt.imshow(src_img.astype(np.uint8))
            # plt.show()
            # src_img = src_img[:,:,:3]
            
            #src_img = np.concatenate((src_img_temp,np.expand_dims(src_img[:,:,3],axis=2)),axis=2)
            #src_img_temp = src_img_temp[:,:,::-1]
            #print('src_img_temp',src_img_temp.shape)
            
            #가중 augmentation
            Augmentation_flag = False
            
            # class 3,5  augmentation하기
            if 3 in GT_img:
                generate_num = 10
                Augmentation_flag = True

            if 5 in GT_img:
                generate_num = 10
                Augmentation_flag = True

            if 7 in GT_img:
                generate_num = 10
                Augmentation_flag = True

            if Augmentation_flag == False:
                continue


            #포함되있는 클래스에 따라 가중치를 줘서 데이터를 늘림
            for i in range(generate_num):
                transformed = transform(image=src_img, mask=GT_img)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

                #print(transformed_image.shape)
                #print(transformed_image.shape)
                #exit(0)

                batch_x.append(transformed_image)
                batch_y.append(transformed_mask)

            # tuple
            Augmentation.save_img(batch_x,file_name = file_name[0] ,path = '',scale = 'RGB')
            Augmentation.save_img(batch_y,file_name = GT_file_name[0] ,path = '',scale = 'GRAY')
        
    #         batch_x.append(transformed_image)
    #         batch_y.append(transformed_mask)

    #         cv2.imshow('test',Visualizer.gray_to_color(transformed_mask))

        return True

    @staticmethod
    def save_img(img_batch ,file_name ,path ,scale= 'RGB'):
        if scale =='RGB':
            folder_name = 'aug_img_patch'
            img_name = file_name[:-4]+'_Aug'
            FGT_tag = ''
        elif scale =='GRAY':
            folder_name = 'aug_label_patch'
            img_name = file_name[:-4]+'_Aug'
            FGT_tag = ''
        else :
            print('scale error scale only input RGB or GRAY')
            return False

        for i in range(len(img_batch)):
            Aug_name = img_name+str(i)+FGT_tag+'.TIF'
            save_path = os.path.join(path,folder_name,Aug_name)
            print('img_batch[i]',img_batch[i])
            print('img_batch[i] shape',img_batch[i].shape)
            print(np.min(img_batch[i]), np.max(img_batch[i]), img_batch[i].shape)
            writeTiff(save_path, img_batch[i], None, None)
            # cv2.imwrite(save_path,img_batch[i].astype(np.uint8))

    #         if scale=='GRAY':
    #             color_save_path = os.path.join(path,'color',Aug_name)
    #             cv2.imwrite(color_save_path,gray_to_color(img_batch[i]))


        return True


transform = A.Compose([
    # A.RandomCrop(width=450, height=450),
    A.RandomSizedCrop(min_max_height=(300, 500), height=512, width=512, p=1),
    A.HorizontalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.5),
    A.PadIfNeeded(min_height=512, min_width=512, p=1)
])


# 경로
x_data_path = ''
y_data_path = ''

img_dirs = {'im_dir': x_data_path, 'label_dir': y_data_path}
img_names = [f for f in os.listdir(img_dirs['im_dir']) if f[-4:] == ".TIF"]
print(img_names)

Augmentation.testing(img_names,img_dirs,transform)
