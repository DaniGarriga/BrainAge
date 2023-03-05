import os
import torch
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict
from utils.logger import get_logger
from utils.utils import absoluteFilePaths

# PARA HACER DATA AUGMENTATION
# SI QUEREMOS USAR ESTARIA BIEN REVISAR PORQUE EN EL PAPER USA COSAS DISTINTAS
from torchvision import transforms 


logger = get_logger('Dataset')


def get_train_loaders(args, augmentation=False, type_data='within', dataset_train='ixi', dataset_test='alfa'):
    logger.info('Creating training and validation set loaders...')

    # OJO: DATA AUG., EN EL PAPER UTILIZAN
    generator_ixi_train = LoadUkbData(mode='train', augmentation=augmentation, type_data=type_data)

    randomlist = random.sample(range(0, len(generator_ixi_train.list_input_total)), len(generator_ixi_train.list_input_total))

    #GENERA LISTAS (70% TRAIN 30% VAL.) CON POSICIONES RANDOM DE 0 A generator_ixi_train.list_input_total (QUE ES EL list_input_total??)
    randomlist_eval = randomlist[round(len(generator_ixi_train.list_input_total) * 0.7):]
    randomlist = randomlist[:round(len(generator_ixi_train.list_input_total) * 0.7)]

    #CON LAS LISTAS ANTERIORES GENERAMOS EL TRAIN Y VAL SET. BATCH SIZE TRIADO EN EL MAIN, PUESTO A 8
    training_generator = DataLoader(generator_ixi_train, batch_size=args.batchSz, sampler=randomlist, num_workers=5)
    val_generator = DataLoader(generator_ixi_train, batch_size=args.batchSz, sampler=randomlist_eval, num_workers=5)

    train_set = training_generator
    val_set = val_generator
    loader = {}
    loader['train'] = train_set
    loader['val'] = val_set

    return loader


# CON ESTA CLASE CARGAMOS LOS DATOS, MUY IMPORTANTE
# PARTE DE LOS METODOS NO SE UTILIZAN, DECIDIR Y BORRAR
class LoadUkbData(Dataset):
    def __init__(self, mode, augmentation=False, type_data=None):

        # ESTO NO SE USA PARA NADA
        '''self.idx_train = [20]
        self.idx_eval = [1]
        self.idx_test = [1]'''
        
        self.mode = mode
        self.augmentation = augmentation

        self.demogs = {}

        self.path_data = os.path.join(os.getcwd(),"ukb_sujetos") # DE AQUI LEEREMOS LOS ARCHIVOS CON LOS DATOS, OJO!

        if augmentation:
            # Flip saggital
            # DIFERENTE PAPER
            transform = transforms.Compose([
                #transforms.Normalize([0], [1]),
                transforms.RandomVerticalFlip(),
                transforms.transforms.RandomAffine(0, translate =(0.01,0.01))
            ])
            self.transform = transform

        #if type_data == 'within':
        #    self.get_random_indexes()

        self.get_labels()
        self.get_list_files()

    def get_list_files(self):

        list_overall_gm_files = absoluteFilePaths(self.path_data) # LISTA CON LOS ARCHIVOS

        lista_imagini = defaultdict()
        lista_mask = defaultdict()
        lista_name = defaultdict()
        lista_outcomes = defaultdict()
        lista_gender = defaultdict()

        ### parse the GM and WM nifty lists for the ones presment in list_extract_subjects ###
        list_parsed_gm = []
        list_parsed_wm = []

        control = 0
        self.info_subjects.set_index("Label", inplace=True)

        for current_subject in self.demogs['subjects']:

            #print(current_subject)
            # try:
            ############################################
            ###### check if there is a nifty file ######
            ############################################

            # MUY IMPORTANTE, PLM GUARDA LOS ARCHIVOS QUE TENGAN EN SU NOMBRE EL ID DEL SUJETO
            plm = [s for s in list_overall_gm_files if str(current_subject) in s]

            if len(plm) > 0:

                list_parsed_gm.append(plm[0]) 
                # EL [0] ESTE DEL FINAL OJO
                # HACE QUE SOLO USEMOS EL PRIMER ARCHIVO QUE CONTENGA EL ID DEL USUARIO
                # CREO QUE PUEDE DAR FALLOS DEPENDE DE COMO ESTE ORGANIZADA LA CARPETAç

                # DEL DATAFRAME, NOS GUARDAMOS EN LAS POSICIONES DEL 0 A len(self.demogs['subjects']) LA EDAD GENERO E ID EN LAS LISTAS ESTAS DE ABAJO
                current_row_of_interest = self.info_subjects.loc[current_subject]
                lista_outcomes[control] = current_row_of_interest.Age
                lista_gender[control] = current_row_of_interest.Genero
                lista_name[control] = current_subject
                control += 1

            elif len(plm) == 0 and control > 0: # SI NO ENCONTRAMOS NINGUN ARCHIVO CON EL ID DEL USUARIO
                lista_outcomes[control] = lista_outcomes[0]
                lista_gender[control] = lista_gender[0]
                lista_name[control] = lista_name[0]
                control += 1

        # ESTO NO LO PILLO TAMPOCO, NO LO COMENTO PORQUE AFECTA A self.list_input_total QUE ES LA LISTA FINAL PERO ?
        list_parsed_gm = [list_parsed_gm[0]] * len(lista_outcomes)
        self.list_parsed_gm = list_parsed_gm

        # NO ENTIENDO PARA QUE CREAMOS EL DIRECTORIO TEMPORAL ESTE TAMPOCO, LO COMENTO DE MOMENTO
        '''from tempfile import mkdtemp
        filename = os.path.join(mkdtemp(), 'newfile.dat')'''

        # ESTO ESTA COMENTADO PORQUE NO SE USA PARA NADA, SUPONGO QUE ERA PARA LEER OTRA COSA
        '''control = 0
        for sth in list_parsed_gm: #range(100): #
            lista_imagini[control] = []
            lista_mask[control] = []
            control += 1'''

        self.list_input_total = list_parsed_gm # MUCHO OJO CON ESTA LISTA PORQUE ES CON LA QUE GENERAMOS LOS SETS
        self.list_output_total = lista_outcomes
        self.ids_total = lista_name

    # ESTE METODO NO SE USA PARA NADA?
    def get_list_files_memory(self, mode):

        list_overall_gm_files = absoluteFilePaths(self.path_data)  # [:1300]
        list_overall_gm_files = [list_overall_gm_files[0]] * 10000

        lista_imagini = defaultdict()
        lista_mask = defaultdict()
        lista_name = defaultdict()
        lista_outcomes = defaultdict()
        lista_gender = defaultdict()

        ### parse the GM and WM nifty lists for the ones presment in list_extract_subjects ###
        list_parsed_gm = []
        list_parsed_wm = []

        control = 0
        self.info_subjects.set_index("Label", inplace=True)

        for current_subject in self.demogs['subjects'][:20000]:

            # print(current_subject)
            # try:
            ############################################
            ###### check if there is a nifty file ######
            ############################################
            plm = [s for s in list_overall_gm_files if str(current_subject) in s]

            if len(plm) > 0:

                list_parsed_gm.append([s for s in list_overall_gm_files if str(current_subject) in s][0])
                current_row_of_interest = self.info_subjects.loc[current_subject]
                lista_outcomes[control] = current_row_of_interest.Age
                # print(current_row_of_interest.Age)
                lista_gender[control] = current_row_of_interest.Genero
                # print(current_row_of_interest.Genero)
                lista_name[control] = current_subject
                control += 1

            elif len(plm) == 0 and control > 0:
                lista_outcomes[control] = lista_outcomes[0]
                lista_gender[control] = lista_gender[0]
                lista_name[control] = lista_name[0]
                control += 1
                # print('Did not have nifty file ')

        # self.get_random_indexes()
        list_parsed_gm = [list_parsed_gm[0]] * len(lista_outcomes)
        self.list_parsed_gm = list_parsed_gm
        from tempfile import mkdtemp

        control = 0
        for sth in list_parsed_gm:  # range(100): #
            lista_imagini[control] = []
            lista_mask[control] = []
            control += 1

        ##### load GM data #####
        control = 0
        control_total = 0
        lista_mask = []
        lista_imagini = []
        self.mmap_input_path = os.path.join(mkdtemp(),'input_data.dat')
        self.mmap_labels_path = os.path.join(mkdtemp(),'labels.dat')
        x = 160
        z = 160
        y = 192
        self.mmap_inputs = None
        self.mmap_labels = None
        for idx, nifty_file in enumerate(list_parsed_gm):
            subject_nifty = absoluteFilePaths(nifty_file)
            mni_subject = [s for s in subject_nifty if 'T1_unbiased_brain' in s]  # 'T1_unbiased_brain_warp'

            if len(mni_subject) > 0:
                label = lista_outcomes[idx]

                if self.mmap_inputs is None:
                    self.mmap_inputs = self.init_mmap(
                        self.mmap_input_path, np.dtype('float32'), (len(list_parsed_gm), * [x,y,z]), remove_existing=True
                    )
                    self.mmap_labels = self.init_mmap(
                        self.mmap_labels_path, label.dtype, (len(list_parsed_gm), *label.shape), remove_existing=True
                    )

                temporar_object = nib.load(mni_subject[0])
                temporar_data = temporar_object.get_fdata()
                temporar_object.uncache()
                # print(temporar_data_gm.shape)
                subject_nifty_fast = absoluteFilePaths(os.path.join(nifty_file, 'T1_fast'))
                mni_subject_mask = [s for s in subject_nifty_fast if 'T1_brain_seg' in s]
                temporar_object_mask = nib.load(mni_subject_mask[0])

                temporar_data_mask = temporar_object_mask.get_fdata()

                # CAMBIAR POR CROP_CENTER
                image, mask = create_crop_mask(np.expand_dims(temporar_data, axis=0).astype('float32'),
                                               np.expand_dims(temporar_data_mask, axis=0).astype('int32'))
                #image = np.stack([image])

                # lista_imagini.append(np.expand_dims(temporar_data, axis=0)) #(temporar_data, axis=-1))
                # lista_mask.append(np.expand_dims(temporar_data_mask, axis=0))  # (temporar_data, axis=-1))
                #lista_imagini.append(image)  # (temporar_data, axis=-1))
                control += 1
                self.mmap_inputs[idx][:] = image
                self.mmap_labels[idx] = label
            else:
                del lista_outcomes[control_total]
                del lista_gender[control_total]
                del lista_name[control_total]

            control_total += 1

        # self.list_input_total =np.stack(list_input_total, axis=0).reshape(-1, list_input_total[0].shape[1],
        #                                                                   list_input_total[0].shape[2],  list_input_total[0].shape[3],
        #                                                                   list_input_total[0].shape[4] )
        self.list_input_total = list_parsed_gm  # lista_imagini #list_input_total_arr
        # self.list_input_total =
        # self.list_output_total = np.stack(list_output_total, axis=0).reshape(-1,list_output_total[0].shape[1],
        #                                                                   list_output_total[0].shape[2],  list_output_total[0].shape[3])
        self.list_output_total = lista_outcomes  # list_output_total
        self.ids_total = lista_name  # names_total
        # self.shape_image_cropped = shape_image

    def get_labels(self):
        PATH_TO_INFO = os.path.join(os.getcwd(),"demographics_icd_new_date3.csv")
        demographics = pd.read_csv(PATH_TO_INFO)
        
        # A PRIORI ESTO ESTA BIEN, SI FALLASE REVISAR
        demographics.loc[:, 'Label'] = demographics['ID'].values.astype(int).astype(str)
        reorder_demogs = demographics.sort_values('Label').reset_index(drop=True)
        self.demogs['age_subjects'] = reorder_demogs['Age'].values
        self.demogs['sex_subjects'] = reorder_demogs['Sex'].values
        name_subjects = reorder_demogs['Label'].values
        self.demogs['subjects'] = name_subjects.tolist() # demogs ES UN DICCIONARIO CON LAS EDADES, SEXO, E ID
        self.info_subjects = reorder_demogs # EN info_subjects GUARDAMOS EL DATAFRAME ENTERO

        #selected_ids = data_demog['ID'].isin(list_ids)
        #self.targets = data_demog['AGE'][selected_ids].values

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_list)
    
    # SOLO SE LLAMA DESDE UN METODO FANTASMA TAMBIEN?
    def init_mmap(self, path, dtype, shape, remove_existing=False): #(self, path: str, dtype: np.dtype, shape: tuple([int]), remove_existing: bool = False) -> np.ndarray:
        open_mode = "r+"

        if remove_existing:
            open_mode = "w+"

        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )

    # NO SE USA
    def __getitem__(self, index):
        #t1_path = self.data_list[index]
        #label = np.array([self.targets[index]])
        nifty_file = self.list_input_total[index] #,:,:,:,:]

        subject_nifty = absoluteFilePaths(nifty_file)
        mni_subject = [s for s in subject_nifty if 'T1_unbiased_brain' in s]  # 'T1_unbiased_brain_warp'
        temporar_object = nib.load(mni_subject[0])
        temporar_data = temporar_object.get_fdata()
        temporar_object.uncache()
        # print(temporar_data_gm.shape)
        subject_nifty_fast = absoluteFilePaths(os.path.join(nifty_file, 'T1_fast'))
        mni_subject_mask = [s for s in subject_nifty_fast if 'T1_brain_seg' in s]
        temporar_object_mask = nib.load(mni_subject_mask[0])
        temporar_data_mask = temporar_object_mask.get_fdata()
        lista_imagini = np.expand_dims(temporar_data, axis=0)  # (temporar_data, axis=-1))
        lista_mask = np.expand_dims(temporar_data_mask, axis=0)  # (temporar_data, axis=-1))
        # Crop background - reduce to image with data
        image, mask = create_crop_mask(lista_imagini.astype('float32'), lista_mask.astype('int32'))
        # list_input_total.append(image.astype('float32'))
        t1 = image

        #t1 = self.mmap_inputs[index]
        # list_input_total_arr[index, :,:,:,:] = image.astype('float32')[0] #np.concatenate([list_input_total_arr, image.astype('float32')[0]])
        #lista_imagini[index] = image.astype('float32')[0]
        # list_mask_total.append(mask)
       # names_total = np.concatenate([names_total, np.array([name])])  # np.array([name]*list_input_cubes.shape[0])])

        #t1 = np.asarray(nib.load(t1_path).dataobj) #.astype(np.float32)
        #list_input_cubes, list_output_cubes = get_3d_patch_voxels(t1, label, 32, 32, image_cov=None)
        label = [self.list_output_total[index]]
       # label = self.mmap_labels[index]
        #subject_id = self.ids_total[index]
        arr_id = self.ids_total[self.ids_total == self.ids_total[index]]

        if self.mode == 'train' and self.augmentation:
            #print('augmentation reee')
            augmented_t1 = self.transform(torch.FloatTensor(t1))

            return torch.FloatTensor(augmented_t1), torch.FloatTensor(label), torch.FloatTensor(label)  # .unsqueeze(0)

        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]

        return torch.FloatTensor(t1), torch.FloatTensor(label), torch.FloatTensor(label)#.unsqueeze(0)


# IGUAL, TODOS ESTOS METODOS HABRIA QUE VALORAR CUALES SON UTILES Y CUALES NO, DE MOMENTO NO SE ESTAN USANDO
def create_crop_mask(image, mask):

    non_zero_image = np.where(image > 0)
    mask_points = defaultdict()
    im_size = image.shape
    x = 160 / 2
    z=160/2
    y = 192/2
    mask_points[0] = [int(round(im_size[1]/2) - x), int(round(im_size[1]/2) + x)] # [np.min(non_zero_image[1])+2, np.max(non_zero_image[1])+2]
    if mask_points[0][1] < 160 or mask_points[0][0] == 0:
        pad_size = 1 + x - int(round(im_size[1]/2))
        image = np.pad(image, ((0, 0), (0, int(pad_size)), (0, 0), (0, 0)), 'constant')
        im_size = image.shape
        mask_points[0] = [int(round(im_size[1] / 2) - x), int(round(im_size[1] / 2) + x)]  # [np.min(non_zero_image[1])+2, np.max(non_zero_image[1])+2]

    mask_points[1] = [int(round(im_size[2]/2) - y), int(round(im_size[2]/2) + y)]
    mask_points[2] = [int(round(im_size[3]/2) - z), int(round(im_size[3]/2) + z)]

    mask_points1 = mask_points[0] #[31, 217]
    mask_points2 = mask_points[1]  #[28, 260]
    mask_points3 = mask_points[2] #[76, 273]

    #cropped_image = image[mask_points[0][0]:mask_points[0][1] , mask_points[1][0]:mask_points[1][1], mask_points[2][0]:mask_points[2][1], :]
    cropped_image = image[: , mask_points1[0]:mask_points1[1], mask_points2[0]:mask_points2[1], mask_points3[0]:mask_points3[1] ]
    cropped_mask = mask[:, mask_points1[0]:mask_points1[1], mask_points2[0]:mask_points2[1], mask_points3[0]:mask_points3[1]]
    if mask_points[0][1] > mask_points1[1] or  mask_points[1][1] > mask_points2[1] or  mask_points[2][1] > mask_points3[1]:
        print('MAX point moved')
        print([mask_points[0], mask_points[1], mask_points[2]])

    if mask_points[0][0] < mask_points1[0] or  mask_points[1][0] < mask_points2[0] or  mask_points[2][0] < mask_points3[0]:
        print('MAX point moved')
        print([mask_points[0], mask_points[1], mask_points[2]])

    return cropped_image, cropped_mask


def get_3d_patch_voxels(image, image_label, mask, block_size_input, block_size_output, image_cov=None):

    # Crop background - reduce to image with data
    image, mask = create_crop_mask(image, mask)

    shape_of_input_data = image.shape
    num_cubes_dim1 = np.int(shape_of_input_data[1] // block_size_output)
    num_cubes_dim2 = np.int(shape_of_input_data[2] // block_size_output)
    num_cubes_dim3 = np.int(shape_of_input_data[3] // block_size_output)

    semi_block_size_input1 = block_size_input//2
    semi_block_size_output1 = block_size_output//2
    semi_block_size_input2 = block_size_input - semi_block_size_input1
    semi_block_size_output2 = block_size_output - semi_block_size_output1

    list_input_cubes = []
    list_output_cubes = []
    list_mask_cubes = []
    min_value_image = np.min(image)

    diff_semi_block1 = semi_block_size_input1 - semi_block_size_output1
    diff_semi_block2 = semi_block_size_input2 - semi_block_size_output2

    print('size of input image going in padding operation')
    print(image.shape)
    #print(diff_semi_block1)
    #print(diff_semi_block2)

    input_image_padded = np.pad(image, ((0, 0), (diff_semi_block1, diff_semi_block2),
                                              (diff_semi_block1, diff_semi_block2),
                                              (diff_semi_block1, diff_semi_block2)
                                              ), mode='constant', constant_values=min_value_image)
    input_mask_padded = np.pad(mask, ((0, 0), (diff_semi_block1, diff_semi_block2),
                                        (diff_semi_block1, diff_semi_block2),
                                        (diff_semi_block1, diff_semi_block2)
                                        ), mode='constant', constant_values=min_value_image)
    #input_image_padded = np.pad(image, ( (diff_semi_block1, diff_semi_block2),
    #                                    (diff_semi_block1, diff_semi_block2),
    #                                    (diff_semi_block1, diff_semi_block2),
    #                                    (0, 0)), mode='constant', constant_values=min_value_image)

    for i in range(num_cubes_dim1):
        for j in range(num_cubes_dim2):
            for k in range(num_cubes_dim3):
                ### extract segmentation space 3D cube ###
                list_output_cubes.append(
                    np.ones((1, block_size_output, block_size_output, block_size_output)) * image_label)
                #print(list_output_cubes[-1].shape)

                ### extract raw input space 3D cube ###
                volumetric_block = input_image_padded[:, block_size_output * i:(block_size_output * i + block_size_input),
                                   block_size_output * j:(block_size_output * j + block_size_input),
                                   block_size_output * k:(block_size_output * k + block_size_input)]
                mask_block = input_mask_padded[:,
                                   block_size_output * i:(block_size_output * i + block_size_input),
                                   block_size_output * j:(block_size_output * j + block_size_input),
                                   block_size_output * k:(block_size_output * k + block_size_input)]
                if image_cov is not None:
                    sex_3d_block = np.ones((1, block_size_input,
                                               block_size_input, block_size_input)) * np.float(image_cov) #np.ones(( block_size_input,
                                               #block_size_input, block_size_input, 1)) * np.float(image_cov)
                    whole_block = np.concatenate((volumetric_block, sex_3d_block), axis=0) #, axis=-1)
                else:
                    whole_block = volumetric_block

                list_input_cubes.append(whole_block)
                list_mask_cubes.append(mask_block)
                #print(np.max(whole_block))
                #print(list_input_cubes[-1].shape)

    list_output_cubes = np.stack(list_output_cubes)
    list_input_cubes = np.stack(list_input_cubes)
    list_mask_cubes = np.stack(list_mask_cubes)

    return list_input_cubes, list_output_cubes, list_mask_cubes, input_image_padded.shape


def stuff_patches_3D(out_shape,patches,xstep=12,ystep=12,zstep=12):
    out = np.zeros(out_shape, patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)
    patch_shape = patches.shape[-3:]
    patches_6D = np.lib.stride_tricks.as_strided(out, ((out.shape[0] - patch_shape[0] + 1) // xstep, (out.shape[1] - patch_shape[1] + 1) // ystep,
                                                  (out.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                  (out.strides[0] * xstep, out.strides[1] * ystep,out.strides[2] * zstep, out.strides[0], out.strides[1],out.strides[2]))
    denom_6D = np.lib.stride_tricks.as_strided(denom, ((denom.shape[0] - patch_shape[0] + 1) // xstep, (denom.shape[1] - patch_shape[1] + 1) // ystep,
                                                  (denom.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                  (denom.strides[0] * xstep, denom.strides[1] * ystep,denom.strides[2] * zstep, denom.strides[0], denom.strides[1],denom.strides[2]))
    np.add.at(patches_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), patches.ravel())
    np.add.at(denom_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), 1)
    return out/denom


def uncubify(arr, oldshape):
    block_size_output = arr.shape[2]
    block_size_input = arr.shape[2]
    num_cubes_dim1 = np.int(oldshape[0] // arr.shape[2])
    num_cubes_dim2 = np.int(oldshape[1] // arr.shape[2])
    num_cubes_dim3 = np.int(oldshape[2] // arr.shape[2])
    reconstructed_image = np.empty(oldshape)
    place = 0
    for i in range(num_cubes_dim1):
        for j in range(num_cubes_dim2):
            for k in range(num_cubes_dim3):
                ### extract segmentation space 3D cube ###
                reconstructed_image[block_size_output * i:(block_size_output * i + block_size_input),
                                   block_size_output * j:(block_size_output * j + block_size_input),
                                   block_size_output * k:(block_size_output * k + block_size_input)] = arr[place]
                place += 1

    ###############################################################
	#### gather smaller 3D blocks into original big 3D block ######
	###############################################################

    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

