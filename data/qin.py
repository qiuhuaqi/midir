class BratsImageDataset(Dataset):
    # Dataloader for Brats dataset during training
    def __init__(self, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.data_path = "..\..\data\Brats\Brats_train"
        # self.data_path = '/int_data009/z003ypkk/COPD_registration/PyTorch-GAN-master/data/Brats_train/train'
        self.filename = [f for f in sorted(os.listdir(self.data_path))]

    def __getitem__(self, index):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        indice = np.random.randint(70, 90)
        patient_path = os.path.join(self.data_path, self.filename[index])
        item_T1 = read_in_mhd_img(os.path.join(patient_path, self.filename[index] + '_t1.nii.gz'))[indice]
        if self.unaligned:
            t2_index = random.randint(0, len(self.filename) - 1)
            patient_path_unaligned = os.path.join(self.data_path, self.filename[t2_index])
            item_T2 = read_in_mhd_img(os.path.join(patient_path_unaligned, self.filename[t2_index] + '_t2.nii.gz'))[
                np.random.randint(70, 90)]
        else:
            item_T2 = read_in_mhd_img(os.path.join(patient_path, self.filename[index] + '_t2.nii.gz'))[indice]
        item_T1 = item_T1[None]
        item_T2 = item_T2[None]
        alpha = np.random.uniform(4000, 5000)
        sigma = 5
        item_T1_tf, flow = elastic_transformations(alpha=alpha, sigma=sigma, rng=np.random.RandomState(
            datetime.datetime.now().second + datetime.datetime.now().microsecond))(item_T1)
        return {'A': item_T1_tf, 'B': item_T2, 'C': flow, 'D': item_T1}

    def __len__(self):
        return len(self.filename)



def rescale_intensity(image, thres=(.05, 99.95)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l+1e-5)
    # image2 = image2 * 2.0 - 1.0
    return image2





def elastic_transformations(alpha, sigma, rng=np.random.RandomState(42),
                            interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    def _elastic_transform_2D(images):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images[0].shape
        mask = np.zeros(image_shape)
        w = int(np.round(image_shape[0]*0.1))
        h = int(np.round(image_shape[1] * 0.15))
        mask[image_shape[0]//2-w:image_shape[0]//2+w, image_shape[1]//2-h:image_shape[1]//2+h] = 1
        # Make random fields
        dx = np.zeros(image_shape, dtype=np.float32)
        dy = np.zeros(image_shape, dtype=np.float32)
        dx[tuple(np.meshgrid(np.arange(0, 240, 30), np.arange(0, 240, 30)))] = rng.uniform(-1, 1, (8, 8)) * alpha
        dy[tuple(np.meshgrid(np.arange(0, 240, 30), np.arange(0, 240, 30)))] = rng.uniform(-1, 1, (8, 8)) * alpha
        brain_mask = (images[0] >= 0.01)
        dx = dx * brain_mask
        dy = dy * brain_mask
        for k in range(5):
            dx = gaussian_filter(dx, sigma=sigma, mode='reflect')
            dy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        flow = np.stack((dx, dy))
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + dy).reshape(-1, 1), \
                            (x + dx).reshape(-1, 1)
        # Map cooordinates from image to distorted index set
        transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                              for image in images]
        transformed_images = np.array(transformed_images, dtype='float32')
        return transformed_images, flow
    return _elastic_transform_2D