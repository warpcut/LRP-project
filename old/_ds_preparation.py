import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio('./urban/test/', output="./urban/DS/", seed=1337, ratio=(.7, .2, .1)) # default values

