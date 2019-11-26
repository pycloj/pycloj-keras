(ns keras.initializers.zero
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

(defn zero 
  "Initializer that generates tensors initialized to 0.
    "
  [  ]
  (py/call-attr initializers "zero"  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
