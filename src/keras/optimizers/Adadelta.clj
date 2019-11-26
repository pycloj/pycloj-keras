(ns keras.optimizers.Adadelta
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
        learning_rate: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.

    # References
        - [Adadelta - an adaptive learning rate method](
           https://arxiv.org/abs/1212.5701)
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

(defn Adadelta 
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
        learning_rate: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.

    # References
        - [Adadelta - an adaptive learning rate method](
           https://arxiv.org/abs/1212.5701)
    "
  [ & {:keys [learning_rate rho]
       :or {learning_rate 1.0 rho 0.95}} ]
  
   (py/call-attr-kw optimizers "Adadelta" [] {:learning_rate learning_rate :rho rho }))

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
