(ns keras.backend.common
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce common (import-module "keras.backend.common"))

(defn cast-to-floatx 
  "Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw common "cast_to_floatx" [] {:x x }))

(defn epsilon 
  "Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    "
  [  ]
  (py/call-attr common "epsilon"   ))

(defn floatx 
  "Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    "
  [  ]
  (py/call-attr common "floatx"   ))

(defn image-data-format 
  "Returns the default image data format convention ('channels_first' or 'channels_last').

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    "
  [  ]
  (py/call-attr common "image_data_format"   ))

(defn image-dim-ordering 
  "Legacy getter for `image_data_format`.

    # Returns
        string, one of `'th'`, `'tf'`
    "
  [  ]
  (py/call-attr common "image_dim_ordering"   ))

(defn normalize-data-format 
  "Checks that the value correspond to a valid data format.

    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    "
  [ & {:keys [value]} ]
   (py/call-attr-kw common "normalize_data_format" [] {:value value }))

(defn set-epsilon 
  "Sets the value of the fuzz factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-07
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    "
  [ & {:keys [e]} ]
   (py/call-attr-kw common "set_epsilon" [] {:e e }))

(defn set-floatx 
  "Sets the default float type.

    # Arguments
        floatx: String, 'float16', 'float32', or 'float64'.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    "
  [ & {:keys [floatx]} ]
   (py/call-attr-kw common "set_floatx" [] {:floatx floatx }))

(defn set-image-data-format 
  "Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```
    "
  [ & {:keys [data_format]} ]
   (py/call-attr-kw common "set_image_data_format" [] {:data_format data_format }))

(defn set-image-dim-ordering 
  "Legacy setter for `image_data_format`.

    # Arguments
        dim_ordering: string. `tf` or `th`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```

    # Raises
        ValueError: if `dim_ordering` is invalid.
    "
  [ & {:keys [dim_ordering]} ]
   (py/call-attr-kw common "set_image_dim_ordering" [] {:dim_ordering dim_ordering }))
