(ns keras.optimizers.Adamax
  "Adamax optimizer from Adam paper's Section 7.

    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
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

(defn Adamax 
  "Adamax optimizer from Adam paper's Section 7.

    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
    "
  [ & {:keys [learning_rate beta_1 beta_2]
       :or {learning_rate 0.002 beta_1 0.9 beta_2 0.999}} ]
  
   (py/call-attr-kw optimizers "Adamax" [] {:learning_rate learning_rate :beta_1 beta_1 :beta_2 beta_2 }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn get-gradients 
  ""
  [ self loss params ]
  (py/call-attr self "get_gradients"  self loss params ))

(defn get-updates 
  ""
  [ self loss params ]
  (py/call-attr self "get_updates"  self loss params ))

(defn get-weights 
  "Returns the current value of the weights of the optimizer.

        # Returns
            A list of numpy arrays.
        "
  [ self  ]
  (py/call-attr self "get_weights"  self  ))

(defn lr 
  ""
  [ self ]
    (py/call-attr self "lr"))

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
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))
