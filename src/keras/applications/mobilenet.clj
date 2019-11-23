(ns keras.applications.mobilenet
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mobilenet (import-module "keras.applications.mobilenet"))

(defn MobileNet 
  ""
  [  ]
  (py/call-attr mobilenet "MobileNet"   ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr mobilenet "decode_predictions"   ))

(defn keras-modules-injection 
  ""
  [ & {:keys [base_fun]} ]
   (py/call-attr-kw mobilenet "keras_modules_injection" [] {:base_fun base_fun }))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr mobilenet "preprocess_input"   ))
