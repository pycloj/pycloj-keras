(ns keras.optimizers.TFOptimizer
  "Wrapper class for native TensorFlow optimizers.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimizers (import-module "keras.optimizers"))

(defn TFOptimizer 
  "Wrapper class for native TensorFlow optimizers.
    "
  [ & {:keys [optimizer]} ]
   (py/call-attr-kw optimizers "TFOptimizer" [] {:optimizer optimizer }))

(defn from-config 
  ""
  [self  & {:keys [config]} ]
    (py/call-attr-kw optimizers "from_config" [self] {:config config }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr optimizers "get_config"  self ))

(defn get-gradients 
  ""
  [self  & {:keys [loss params]} ]
    (py/call-attr-kw optimizers "get_gradients" [self] {:loss loss :params params }))

(defn get-updates 
  ""
  [self  & {:keys [loss params]} ]
    (py/call-attr-kw optimizers "get_updates" [self] {:loss loss :params params }))

(defn get-weights 
  "Returns the current value of the weights of the optimizer.

        # Returns
            A list of numpy arrays.
        "
  [ self ]
  (py/call-attr optimizers "get_weights"  self ))

(defn set-weights 
  "Sets the weights of the optimizer, from Numpy arrays.

        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: in case of incompatible weight shapes.
        "
  [self  & {:keys [weights]} ]
    (py/call-attr-kw optimizers "set_weights" [self] {:weights weights }))

(defn weights 
  ""
  [ self ]
    (py/call-attr optimizers "weights"  self))
