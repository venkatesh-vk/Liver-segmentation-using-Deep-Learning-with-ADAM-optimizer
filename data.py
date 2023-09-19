import os
import numpy as np
import nibabel

data_path = 'raw/'
image_rows = int(512/2)
image_cols = int(512/2) 


def create_train_data():
    print('-'*30)
    print('Creating training data...')
    print('-'*30)
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    imgs_train = [] 
    masks_train = []    
    training_masks = images[::2]

    training_images = images[1::2] 
        
    for liver, orig in zip(training_masks, training_images):
        training_mask = nibabel.load(os.path.join(train_data_path, liver))
        training_image = nibabel.load(os.path.join(train_data_path, orig)) 
        
        for k in range(training_mask.shape[2]):
            mask_2d = np.array(training_mask.get_data()[::2, ::2, k]) 
            image_2d = np.array(training_image.get_data()[::2, ::2, k])
            if len(np.unique(mask_2d)) != 1:
                masks_train.append(mask_2d)
                imgs_train.append(image_2d)
                    
    imgs = np.ndarray(
            (len(imgs_train), image_rows, image_cols), dtype=np.uint8
            )
    imgs_mask = np.ndarray(
            (len(masks_train), image_rows, image_cols), dtype=np.uint8
            )
    
    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img
        
    for index, img in enumerate(masks_train):
        imgs_mask[index, :, :] = img

    np.save('imgs_train.npy', imgs)
    np.save('masks_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    masks_train = np.load('masks_train.npy')
    return imgs_train, masks_train


def create_test_data():
    print('-'*30)
    print('Creating test data...')
    print('-'*30)
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)   
    imgs_test = []
    masks_test = []
    
    for image_name in images:
        print(image_name)
        img = nibabel.load(os.path.join(test_data_path, image_name))
        print(img.shape)
        
        for k in range(img.shape[2]):  
            img_2d = np.array(img.get_data()[::2, ::2, k])
            
            if 'liver' in image_name:
                masks_test.append(img_2d)
            
            elif 'orig' in image_name:
                imgs_test.append(img_2d)
                      
    imgst = np.ndarray(
            (len(imgs_test), image_rows, image_cols), dtype=np.uint8
            )
    imgs_maskt = np.ndarray(
            (len(masks_test), image_rows, image_cols), dtype=np.uint8
            )
    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img
        
    for index, img in enumerate(masks_test):
        imgs_maskt[index, :, :] = img

    np.save('imgs_test.npy', imgst)
    np.save('masks_test.npy', imgs_maskt)
    print('Saving to .npy files done.')
    

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    masks_test = np.load('masks_test.npy')
    return imgs_test, masks_test


if __name__ == '__main__':
    # create_train_data()
    # create_test_data()
    a="""
    ------------------------------
    Creating training data...
    ------------------------------
    Saving to .npy files done.
    ------------------------------
    Creating test data...
    ------------------------------
    ircad_e14_liver.nii.gz
    (512, 512, 113)
    ircad_e14_orig.nii.gz
    (512, 512, 113)
    ircad_e15_liver.nii.gz
    (512, 512, 125)
    ircad_e15_orig.nii.gz
    (512, 512, 125)
    ircad_e16_liver.nii.gz
    (512, 512, 155)
    ircad_e16_orig.nii.gz
    (512, 512, 155)
    ircad_e17_liver.nii.gz
    (512, 512, 119)
    ircad_e17_orig.nii.gz
    (512, 512, 119)
    ircad_e18_liver.nii.gz
    (512, 512, 74)
    ircad_e18_orig.nii.gz
    (512, 512, 74)
    ircad_e19_liver.nii.gz
    (512, 512, 124)
    ircad_e19_orig.nii.gz
    (512, 512, 124)
    ircad_e20_liver.nii.gz
    (512, 512, 225)
    ircad_e20_orig.nii.gz
    (512, 512, 225)
    Saving to .npy files done.
    """
    print(a)