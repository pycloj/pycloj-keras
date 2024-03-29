(ns keras.initializers.Initializer
  "Initializer base class: all initializers inherit from this class.
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

(defn Initializer 
  "Initializer base class: all initializers inherit from this class.
    "
  [  ]
  (py/call-attr initializers "Initializer"  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
