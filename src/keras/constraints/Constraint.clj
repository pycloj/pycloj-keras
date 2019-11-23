(ns keras.constraints.Constraint
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "keras.constraints"))

(defn Constraint 
  ""
  [  ]
  (py/call-attr constraints "Constraint"   ))

(defn get-config 
  ""
  [ self ]
  (py/call-attr constraints "get_config"  self ))
