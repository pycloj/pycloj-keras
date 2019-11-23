(ns keras.optimizers.nadam
  "Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning]
          (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
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

(defn nadam 
  "Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning]
          (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    "
  [ & {:keys [lr beta_1 beta_2 epsilon schedule_decay]
       :or {lr 0.002 beta_1 0.9 beta_2 0.999 schedule_decay 0.004}} ]
  
   (py/call-attr-kw optimizers "nadam" [] {:lr lr :beta_1 beta_1 :beta_2 beta_2 :epsilon epsilon :schedule_decay schedule_decay }))

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
