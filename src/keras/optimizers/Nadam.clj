(ns keras.optimizers.Nadam
  "Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
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

(defn Nadam 
  "Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    "
  [ & {:keys [learning_rate beta_1 beta_2]
       :or {learning_rate 0.002 beta_1 0.9 beta_2 0.999}} ]
  
   (py/call-attr-kw optimizers "Nadam" [] {:learning_rate learning_rate :beta_1 beta_1 :beta_2 beta_2 }))

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
