(ns keras.initializers.random-uniform
  "Initializer that generates tensors with a uniform distribution.

    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "keras.initializers"))

(defn random-uniform 
  "Initializer that generates tensors with a uniform distribution.

    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    "
  [ & {:keys [minval maxval seed]
       :or {minval -0.05 maxval 0.05}} ]
  
   (py/call-attr-kw initializers "random_uniform" [] {:minval minval :maxval maxval :seed seed }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr initializers "get_config"  self ))
