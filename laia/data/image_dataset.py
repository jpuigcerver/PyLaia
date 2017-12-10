from torch.utils import data
from PIL import Image

class ImageDataset(data.Dataset):
    def __init__(self, imgs, transform=None):
        assert isinstance(imgs, (list, tuple))
        super(ImageDataset, self).__init__()
        self._imgs = imgs
        self._transform = transform

    def __getitem__(self, index):
        """Returns a dictionary contaning the given image from the dataset.
        The image is associated with the key 'img'."""
        img = Image.open(self._imgs[index])
        if self._transform:
            img = self._transform(img)
        return {'img': img}

    def __len__(self):
        return len(self._imgs)
