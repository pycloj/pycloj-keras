(ns keras.optimizers.Adagrad
  "Adagrad optimizer.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the learning rate.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Initial learning rate.

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
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

(defn Adagrad 
  "Adagrad optimizer.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the learning rate.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Initial learning rate.

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    "
  [ & {:keys [learning_rate]
       :or {learning_rate 0.01}} ]
  
   (py/call-attr-kw optimizers "Adagrad" [] {:learning_rate learning_rate }))

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
  ""
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))
