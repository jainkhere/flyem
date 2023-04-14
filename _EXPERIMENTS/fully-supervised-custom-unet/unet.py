from imports import *
import metrics
import data_management
import interactive_plot

def load_model_unet(file, use_dice_loss=False, use_jaccard_loss=False, use_focal_loss=False):
    from keras.models import load_model

    cobj = {'keras_precision': metrics.keras_precision, 'keras_recall': metrics.keras_recall, 'keras_jaccard_coef': metrics.keras_jaccard_coef, 'keras_dice_coef': metrics.keras_dice_coef, 'iou': iou, 'iou_thresholded': iou_thresholded, 'accuracy': 'accuracy'}
    if use_dice_loss:
        cobj['keras_dice_coef_loss'] = metrics.keras_dice_coef_loss
    if use_jaccard_loss:
        cobj['keras_jaccard_distance_loss'] = metrics.keras_jaccard_distance_loss
    if use_focal_loss:
        cobj['keras_focal_loss'] = metrics.keras_focal_loss
    model = keras.models.load_model(file, custom_objects=cobj)
    return model

def predict_net(model, img, verbose=1):
    imgs_mask_test = model.predict(img, batch_size=1, verbose=verbose)
    return imgs_mask_test

def train(model, imgs_train, imgs_mask_train, imgs_test, imgs_mask_test, model_path, bt_size, train_epochs, iter_per_epoch=100, val_steps=40, finetune_path=None, perform_centering=False, perform_flipping=True, perform_rotation=True, perform_standardization=False, plot_graph=True, verbosity=1, to_dir=False, train_on_borders=False):
      
    print('-------------------------------')
    print('data details:')

    print('imgs_train.shape', imgs_train.shape)
    print('imgs_mask_train.shape', imgs_mask_train.shape)
    print('imgs_test.shape', imgs_test.shape)
    print('imgs_mask_test.shape', imgs_mask_test.shape)
    print('imgs_train.dtype', imgs_train.dtype)
    print('imgs_mask_train.dtype', imgs_mask_train.dtype)
    print('imgs_test.dtype', imgs_test.dtype)
    print('imgs_mask_test.dtype', imgs_mask_test.dtype)
    
    print('balance:')
    totalpx = (imgs_mask_train.shape[0]*imgs_mask_train.shape[1]*imgs_mask_train.shape[2])
    marked = np.sum(imgs_mask_train>0.5)
    print('train_total_px', totalpx)
    print('train_labeled_positive', marked)
    resag = marked/float(totalpx)
    print('train_fraction_positive', resag)
    print('min, max train', imgs_train.min(), imgs_train.max())
    print('train_total_px', totalpx, 'number 0/1 in mask', np.sum(imgs_mask_train==0)+np.sum(imgs_mask_train==1))

    totalpx_test = (imgs_mask_test.shape[0]*imgs_mask_test.shape[1]*imgs_mask_test.shape[2])
    marked_test = np.sum(imgs_mask_test>0.5)
    print('test_total_px', totalpx_test)
    print('test_labeled_positive', marked_test)
    resag_test = marked_test/float(totalpx_test)
    print('test_fraction_positive', resag_test)        
    print('min, max test', imgs_test.min(), imgs_test.max())
    
    print("loading data done")
        
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    sname = model_path + 'weights.{epoch:02d}-{loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(sname, monitor='val_loss',verbose=1, save_best_only=False, save_freq=5)
    print('saving model checkpoints in', model_path)
    
    logfile_path = model_path + 'log.txt'
    lcallbacks = [interactive_plot.InteractivePlot(logfile_path, plot_graph, iter_per_epoch), model_checkpoint]
    print('saving logfile in ', str(logfile_path))
    
    
    data_gen_args = dict()
    data_val_gen_args = dict()
    print('augmenting on the fly...')
    print('computing statistics...')
    train_mean =None
    train_sd = None
    
    if perform_standardization:
        print('will standardize!')
        train_mean = np.mean(imgs_train, axis=0)
        train_sd = np.std(imgs_train, axis=0)
    elif perform_centering:
        print('will center!')
        train_mean = np.mean(imgs_train, axis=0)

    def cus_raw(img):
        return data_management.custom_preproc(img, 'raw', flip_z=perform_flipping, rotate=perform_rotation, mean=train_mean, sd=train_sd, train_on_borders=train_on_borders)

    def cus_mask(img):
        return data_management.custom_preproc(img, 'mask', flip_z=perform_flipping, rotate=perform_rotation, mean=train_mean, sd=train_sd, train_on_borders=train_on_borders)

    seed = 1
    
    #setting up train generator
    data_gen_args = dict(num_channels = MEMORY,
                         featurewise_center=False,
                         featurewise_std_normalization=False,               
                         samplewise_center=False, 
                         samplewise_std_normalization=False,
                         #rotation_range=180.,
                         #width_shift_range=0.05, 
                         #height_shift_range=0.05, 
                         #channel_shift_range=0.05,
                         #fill_mode='constant', cval=0,
                         preprocessing_function = cus_raw,               
                         horizontal_flip=perform_flipping,
                         vertical_flip=perform_flipping)

    datagen = extension.ExtImageDataGenerator(**data_gen_args) 
    
    data_gen_args_mask = data_gen_args.copy()
    data_gen_args_mask['num_channels'] = 1 
    data_gen_args_mask['preprocessing_function'] = cus_mask 
    data_gen_args_mask['featurewise_center'] = False
    data_gen_args_mask['featurewise_std_normalization'] = False
    data_gen_args_mask['samplewise_center'] = False
    data_gen_args_mask['samplewise_std_normalization'] = False
    datagen_mask = extension.ExtImageDataGenerator(**data_gen_args_mask)

    datagen.fit(imgs_train, augment=False, seed=seed)
    datagen_mask.fit(imgs_mask_train, augment=False, seed=seed)

    
    train_dir = None
    test_dir = None
    train_pref = None
    test_pref = None
    if to_dir:
        train_dir = 'a'
        test_dir = 'b'
        train_pref = 'img'
        test_pref = 'img'
        print('saving augmented images!')
    datagen = datagen.flow(imgs_train, batch_size=bt_size,shuffle=True, seed=seed, save_to_dir=train_dir, save_prefix=train_pref)
    datagen_mask = datagen_mask.flow(imgs_mask_train, batch_size=bt_size, shuffle=True,seed=seed, save_to_dir=test_dir, save_prefix=test_pref)

    def combine_generator(gen1, gen2):
        while True:
            yield(next(gen1), next(gen2))
    
    train_generator = combine_generator(datagen, datagen_mask)#zip(datagen, datagen_mask)

    #setting up validation generator
    data_val_gen_args = dict(num_channels = MEMORY,
                             featurewise_center=False,
                             featurewise_std_normalization=False,
                             preprocessing_function = cus_raw,
                             samplewise_center=False,
                             samplewise_std_normalization=False)

    datagen_val = extension.ExtImageDataGenerator(**data_val_gen_args)
   
    data_val_gen_args_mask = data_val_gen_args.copy()
    data_val_gen_args_mask['num_channels'] = 1 
    data_val_gen_args_mask['preprocessing_function'] = cus_mask 
    data_val_gen_args_mask['featurewise_center'] = False
    data_val_gen_args_mask['featurewise_std_normalization'] = False
    data_val_gen_args_mask['samplewise_center'] = False
    data_val_gen_args_mask['samplewise_std_normalization'] = False
    datagen_val_mask = extension.ExtImageDataGenerator(**data_val_gen_args_mask)
    
    datagen_val.fit(imgs_train, augment=False, seed=seed) #fit to train, to substract training and not test mean
    datagen_val_mask.fit(imgs_mask_train, augment=False, seed=seed)

    datagen_val = datagen_val.flow(imgs_test, batch_size=bt_size, shuffle=True, seed=seed)
    datagen_val_mask = datagen_val_mask.flow(imgs_mask_test, batch_size=bt_size, shuffle=True, seed=seed)

    test_generator = combine_generator(datagen_val, datagen_val_mask)#zip(datagen_val, datagen_val_mask)

    print(train_generator)
    model.fit_generator(train_generator,
                             steps_per_epoch=iter_per_epoch, epochs=train_epochs, 
                             validation_data=test_generator,
                             validation_steps=val_steps, callbacks=lcallbacks, verbose=verbosity)

def get_unet(lrate, diceloss=False, jaccardloss=False, focalloss=False, customloss=False):
    lss = 'binary_crossentropy' 
    if diceloss:
        print('using dice loss!')
        lss = metrics.keras_dice_coef_loss
    elif jaccardloss:
        print('using jaccard loss!')
        lss = metrics.keras_jaccard_distance_loss
    elif focalloss:
        print('using focal loss!')
        lss = metrics.keras_focal_loss
    elif customloss:
        lss = metrics.keras_binary_crossentropy_mod

    model = custom_unet(
        input_shape=(512, 512, 1),
        use_batch_norm=False,
        num_classes=1,
        filters=32,
        dropout=0.5,
        output_activation='sigmoid')

    model.compile(optimizer = Adam(lr = lrate), loss = lss, metrics = ['accuracy', iou, iou_thresholded, metrics.keras_precision, metrics.keras_recall, metrics.keras_jaccard_coef, metrics.keras_dice_coef])

    return model


#normal execute_predict
def execute_predict(model, img_in, stepsize=512, resize_shortest=True, extensive=True):
    return model.predict(img_in) 

def improve_components(test_pred, depth=9):
    return median_filter(test_pred, size=(depth,1,1,1)) #run z-smoothing (median filter)
   
def predict(model, img, groundtruth=None, overlay=True, threshold=0.1, stepsize=512, resize_shortest=True, verbose=False):
    img_in = img.astype(float)
    img_in/=255.

    newimg = np.zeros((img.shape[1], img.shape[2], 3))
    newimg[:,:,0] = img[0,:,:,0]
    newimg[:,:,1] = img[0,:,:,0]
    newimg[:,:,2] = img[0,:,:,0]   
    
    res = execute_predict(model, img_in, stepsize, resize_shortest)
    if verbose:
        print('res', res.shape)
        print('res_min', res.min())
        print('res_max', res.max())
        print('part pos.', np.sum(res>threshold))
    
    tres = res[0,:,:,0]
    tres = (1-tres)
    tres_embedded = np.zeros((tres.shape[0], tres.shape[1], 4))
    tres_embedded[:,:,0] = 255
    tres_embedded[:,:,1] = tres*0
    tres_embedded[:,:,2] = tres*0
    tres_embedded[:,:,3] = tres>threshold
    
    blended = tres_embedded
    only_gt = tres_embedded
    only_pred = tres_embedded
    if overlay:
        blended = newimg.copy() 
        only_gt = newimg.copy()
        only_pred = newimg.copy()
        if groundtruth is not None:
            print('gt', groundtruth.shape)
            only_gt[groundtruth[:,:,0]>0] = np.minimum(only_gt[groundtruth[:,:,0]>0]+(0,100,0), 255) #(0,255,0) green for groundtruth
            blended[groundtruth[:,:,0]>0] = np.minimum(blended[groundtruth[:,:,0]>0]+(0,100,0), 255) #(0,255,0) green for groundtruth
        
        only_pred[tres<threshold] = np.minimum(only_pred[tres<threshold]+(100,0,0), 255) #(255,0,0) red for detections
        blended[tres<threshold] = np.minimum(blended[tres<threshold]+(100,0,0), 255) #(255,0,0) red for detections
        
        blended = (blended-blended.min())/(blended.max()-blended.min())
        only_pred = (only_pred-only_pred.min())/(only_pred.max()-only_pred.min())
        only_gt = (only_gt-only_gt.min())/(only_gt.max()-only_gt.min())
        
    return blended, tres_embedded, res, only_pred, only_gt

def eval_image(input_img, gt, stepsize=512, resize_shortest=True, verbose=False):
    if not gt is None:
        gt = gt.copy()
        gt[gt==255]=1
        
    if verbose:
        print('input_img.shape', input_img.shape)
        print(input_img.min(), input_img.max())
    plt.figure(figsize=(10,10))
    plt.imshow(input_img, cmap='gray')

    testinp = input_img.reshape(1,input_img.shape[0],input_img.shape[1],1)

    if verbose:
        print(testinp.shape)
        print('min', testinp.min(), 'max', testinp.max())
    blended, tres_embedded, res, only_pred, only_gt = predict(testinp, gt, threshold=0.5, stepsize=stepsize, resize_shortest=resize_shortest, verbose=verbose)
    plt.figure(figsize=(10,10))
    plt.imshow(blended)
    
    #color overlay
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('prediction')
    plt.imshow(only_pred)

    plt.subplot(1,2,2)
    plt.title('groundtruth')
    plt.imshow(only_gt) 
    
    #binary
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('prediction')
    plt.imshow(tres_embedded, cmap='gray')

    if not gt is None:
        plt.subplot(1,2,2)
        plt.title('groundtruth')
        plt.imshow(gt[:,:,1], cmap='gray')
    
    res[res>0.5] = 1
    res[res<=0.5] = 0
    res = res.astype(np.uint8)
    if not gt is None:
        print('jaccard', jaccard_index_single(res[0,:,:,0], gt[:,:,0], verbose=verbose)[0])
