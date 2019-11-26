(ns keras.initializers.Constant
  "Initializer that generates tensors initialized to a constant value.

    # Arguments
        value: float; the value of the generator tensors.
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

(defn Constant 
  "Initializer that generates tensors initialized to a constant value.

    # Arguments
        value: float; the value of the generator tensors.
    "
  [ & {:keys [value]
       :or {value 0}} ]
  
   (py/call-attr-kw initializers "Constant" [] {:value value }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
