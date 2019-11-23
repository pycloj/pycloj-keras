(ns keras.applications.xception
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce xception (import-module "keras.applications.xception"))

(defn Xception 
  ""
  [  ]
  (py/call-attr xception "Xception"   ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr xception "decode_predictions"   ))

(defn keras-modules-injection 
  ""
  [ & {:keys [base_fun]} ]
   (py/call-attr-kw xception "keras_modules_injection" [] {:base_fun base_fun }))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr xception "preprocess_input"   ))
