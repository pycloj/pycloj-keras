(ns keras.initializers.Ones
  "Initializer that generates tensors initialized to 1.
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

(defn Ones 
  "Initializer that generates tensors initialized to 1.
    "
  [  ]
  (py/call-attr initializers "Ones"  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
