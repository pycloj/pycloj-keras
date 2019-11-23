(ns keras.applications.vgg16
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce vgg16 (import-module "keras.applications.vgg16"))

(defn VGG16 
  ""
  [  ]
  (py/call-attr vgg16 "VGG16"   ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr vgg16 "decode_predictions"   ))

(defn keras-modules-injection 
  ""
  [ & {:keys [base_fun]} ]
   (py/call-attr-kw vgg16 "keras_modules_injection" [] {:base_fun base_fun }))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr vgg16 "preprocess_input"   ))
