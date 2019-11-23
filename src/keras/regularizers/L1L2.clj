(ns keras.regularizers.L1L2
  "Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce regularizers (import-module "keras.regularizers"))

(defn L1L2 
  "Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    "
  [ & {:keys [l1 l2]
       :or {l1 0.0 l2 0.0}} ]
  
   (py/call-attr-kw regularizers "L1L2" [] {:l1 l1 :l2 l2 }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr regularizers "get_config"  self ))
