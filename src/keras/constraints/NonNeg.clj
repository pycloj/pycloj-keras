(ns keras.constraints.nonneg
  "Constrains the weights to be non-negative.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "keras.constraints"))

(defn nonneg 
  "Constrains the weights to be non-negative.
    "
  [  ]
  (py/call-attr constraints "nonneg"   ))

(defn get-config 
  ""
  [ self ]
  (py/call-attr constraints "get_config"  self ))
