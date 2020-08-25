from torch.utils.data import DataLoader

from laia.data.padding_collater import PaddingCollater, by_descending_width


class ImageDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        image_channels=1,
        image_height=None,
        image_width=None,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=PaddingCollater(
                {"img": [image_channels, image_height, image_width]},
                sort_key=by_descending_width,
            ),
            worker_init_fn=worker_init_fn,
        )
