(ns keras.applications.nasnet
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce nasnet (import-module "keras.applications.nasnet"))

(defn NASNetLarge 
  ""
  [  ]
  (py/call-attr nasnet "NASNetLarge"   ))

(defn NASNetMobile 
  ""
  [  ]
  (py/call-attr nasnet "NASNetMobile"   ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr nasnet "decode_predictions"   ))

(defn keras-modules-injection 
  ""
  [ & {:keys [base_fun]} ]
   (py/call-attr-kw nasnet "keras_modules_injection" [] {:base_fun base_fun }))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr nasnet "preprocess_input"   ))
