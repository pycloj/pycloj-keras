(ns keras.initializers.zeros
  "Initializer that generates tensors initialized to 0.
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

(defn zeros 
  "Initializer that generates tensors initialized to 0.
    "
  [  ]
  (py/call-attr initializers "zeros"   ))

(defn get-config 
  ""
  [ self ]
  (py/call-attr initializers "get_config"  self ))
