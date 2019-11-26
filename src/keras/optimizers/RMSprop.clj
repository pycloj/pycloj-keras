(ns keras.optimizers.RMSprop
  "RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    # Arguments
        learning_rate: float >= 0. Learning rate.
        rho: float >= 0.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
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

(defn RMSprop 
  "RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    # Arguments
        learning_rate: float >= 0. Learning rate.
        rho: float >= 0.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    "
  [ & {:keys [learning_rate rho]
       :or {learning_rate 0.001 rho 0.9}} ]
  
   (py/call-attr-kw optimizers "RMSprop" [] {:learning_rate learning_rate :rho rho }))

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
