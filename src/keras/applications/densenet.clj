(ns keras.applications.densenet
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce densenet (import-module "keras.applications.densenet"))

(defn DenseNet121 
  ""
  [  ]
  (py/call-attr densenet "DenseNet121"  ))

(defn DenseNet169 
  ""
  [  ]
  (py/call-attr densenet "DenseNet169"  ))

(defn DenseNet201 
  ""
  [  ]
  (py/call-attr densenet "DenseNet201"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr densenet "decode_predictions"  ))

(defn keras-modules-injection 
  ""
  [ base_fun ]
  (py/call-attr densenet "keras_modules_injection"  base_fun ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr densenet "preprocess_input"  ))
