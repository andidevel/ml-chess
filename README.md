# Chess Pieces Identification

This is the repository for the class `EEL7513 - Machine Learning` project.

## HDF5 Data

You need to build your own dataset running `build_image_dataset` script. After running the script
a file `chess_dataset.h5` will be on `data` folder.

The HDF5 file created after run `build_image_dataset` script has the following structure:

- `chess_imgs_train`: samples dataset to train, with shape (n, 227, 227, 3) where `n` is the number of samples.
- `chess_labels_train`: chess_imgs_train labels with shape (n, 2) where `n` is the number of samples.
- `chess_imgs_test`: samples dataset with shape (n, 227, 227, 3) where `n` is the number of samples.
- `chess_labels_test`: chess_imgs_test labels with shape (n, 2) where `n` is the number of samples.

**chess_labels example**

If chess_imgs_train[n] is a black king piece, so chess_labels_train[n] == ['1', 'bk'], as well `chess_imgs_test` and `chess_labels_test`.

# Chess Labels

```python
labels = [
    'bb',
    'bk',
    'bn',
    'bp',
    'bq',
    'br',
    'empty',
    'wb',
    'wk',
    'wn',
    'wp',
    'wq',
    'wr'
]
```
