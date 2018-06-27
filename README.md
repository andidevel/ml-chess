# Chess Pieces Identification

This is the repository for the class `EEL7513 - Machine Learning` project.

## HDF5 Data

You need to build your own dataset running `build_image_dataset` script.

The HDF5 file created after run `build_image_dataset` script has the following structure:

- `chess_imgs`: samples dataset with shape (n, 227, 227, 3) where `n` is the number of samples.
- `chess_labels`: chess_imgs labels with shape (n, 2) where `n` is the number of samples.

The HDF5 file will be on the `data` folder.

**chess_labels example**

If chess_imgs[n] is a black king piece, so chess_labels[n] == ['1', 'bk']

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
