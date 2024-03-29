(ns keras.regularizers.Regularizer
  "Regularizer base class.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce regularizers (import-module "keras.regularizers"))

(defn Regularizer 
  "Regularizer base class.
    "
  [  ]
  (py/call-attr regularizers "Regularizer"  ))
