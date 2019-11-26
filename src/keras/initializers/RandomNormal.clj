(ns keras.initializers.RandomNormal
  "Initializer that generates tensors with a normal distribution.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
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

(defn RandomNormal 
  "Initializer that generates tensors with a normal distribution.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    "
  [ & {:keys [mean stddev seed]
       :or {mean 0.0 stddev 0.05}} ]
  
   (py/call-attr-kw initializers "RandomNormal" [] {:mean mean :stddev stddev :seed seed }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
