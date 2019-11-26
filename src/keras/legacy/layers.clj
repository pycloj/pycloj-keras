(ns keras.legacy.layers
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce layers (import-module "keras.legacy.layers"))

(defn AtrousConvolution1D 
  ""
  [  ]
  (py/call-attr layers "AtrousConvolution1D"  ))

(defn AtrousConvolution2D 
  ""
  [  ]
  (py/call-attr layers "AtrousConvolution2D"  ))

(defn to-list 
  "Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    "
  [x & {:keys [allow_tuple]
                       :or {allow_tuple false}} ]
    (py/call-attr-kw layers "to_list" [x] {:allow_tuple allow_tuple }))
