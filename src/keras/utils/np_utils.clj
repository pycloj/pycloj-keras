(ns keras.utils.np-utils
  "Numpy-related utilities."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce np-utils (import-module "keras.utils.np_utils"))

(defn normalize 
  "Normalizes a Numpy array.

    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).

    # Returns
        A normalized copy of the array.
    "
  [ & {:keys [x axis order]
       :or {axis -1 order 2}} ]
  
   (py/call-attr-kw np-utils "normalize" [] {:x x :axis axis :order order }))

(defn to-categorical 
  "Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    "
  [ & {:keys [y num_classes dtype]
       :or {dtype "float32"}} ]
  
   (py/call-attr-kw np-utils "to_categorical" [] {:y y :num_classes num_classes :dtype dtype }))
