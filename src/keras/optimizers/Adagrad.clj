(ns keras.optimizers.adagrad
  "Adagrad optimizer.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Initial learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.

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

(defn adagrad 
  "Adagrad optimizer.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Initial learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    "
  [ & {:keys [lr epsilon decay]
       :or {lr 0.01 decay 0.0}} ]
  
   (py/call-attr-kw optimizers "adagrad" [] {:lr lr :epsilon epsilon :decay decay }))

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
