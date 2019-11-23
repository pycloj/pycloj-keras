(ns keras.optimizers.adadelta
  "Adadelta optimizer.

    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Initial learning rate decay.

    # References
        - [Adadelta - an adaptive learning rate method]
          (https://arxiv.org/abs/1212.5701)
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

(defn adadelta 
  "Adadelta optimizer.

    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Initial learning rate decay.

    # References
        - [Adadelta - an adaptive learning rate method]
          (https://arxiv.org/abs/1212.5701)
    "
  [ & {:keys [lr rho epsilon decay]
       :or {lr 1.0 rho 0.95 decay 0.0}} ]
  
   (py/call-attr-kw optimizers "adadelta" [] {:lr lr :rho rho :epsilon epsilon :decay decay }))

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
