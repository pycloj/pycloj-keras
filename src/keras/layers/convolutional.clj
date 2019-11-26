(ns keras.layers.convolutional
  "Convolutional layers.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce convolutional (import-module "keras.layers.convolutional"))

(defn AtrousConv1D 
  ""
  [  ]
  (py/call-attr convolutional "AtrousConv1D"  ))

(defn AtrousConv2D 
  ""
  [  ]
  (py/call-attr convolutional "AtrousConv2D"  ))

(defn AtrousConvolution1D 
  ""
  [  ]
  (py/call-attr convolutional "AtrousConvolution1D"  ))

(defn AtrousConvolution2D 
  ""
  [  ]
  (py/call-attr convolutional "AtrousConvolution2D"  ))

(defn transpose-shape 
  "Converts a tuple or a list to the correct `data_format`.

    It does so by switching the positions of its elements.

    # Arguments
        shape: Tuple or list, often representing shape,
            corresponding to `'channels_last'`.
        target_format: A string, either `'channels_first'` or `'channels_last'`.
        spatial_axes: A tuple of integers.
            Correspond to the indexes of the spatial axes.
            For example, if you pass a shape
            representing (batch_size, timesteps, rows, cols, channels),
            then `spatial_axes=(2, 3)`.

    # Returns
        A tuple or list, with the elements permuted according
        to `target_format`.

    # Example
    ```python
        >>> from keras.utils.generic_utils import transpose_shape
        >>> transpose_shape((16, 128, 128, 32),'channels_first', spatial_axes=(1, 2))
        (16, 32, 128, 128)
        >>> transpose_shape((16, 128, 128, 32), 'channels_last', spatial_axes=(1, 2))
        (16, 128, 128, 32)
        >>> transpose_shape((128, 128, 32), 'channels_first', spatial_axes=(0, 1))
        (32, 128, 128)
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    "
  [ shape target_format spatial_axes ]
  (py/call-attr convolutional "transpose_shape"  shape target_format spatial_axes ))
