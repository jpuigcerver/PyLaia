from torch.utils import data
from PIL import Image

class ImageDataset(data.Dataset):
    def __init__(self, imgs, transform=None):
        assert isinstance(imgs, (list, tuple))
        super(ImageDataset, self).__init__()
        self._imgs = imgs
        self._transform = transform

    def __getitem__(self, index):
        """Returns an image from the dataset."""
        img = Image.open(self._imgs[index])
        if self._transform:
            img = self._transform(img)
        return img

    def __len__(self):
        return len(self._imgs)
