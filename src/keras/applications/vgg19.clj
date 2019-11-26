(ns keras.applications.vgg19
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce vgg19 (import-module "keras.applications.vgg19"))

(defn VGG19 
  ""
  [  ]
  (py/call-attr vgg19 "VGG19"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr vgg19 "decode_predictions"  ))

(defn keras-modules-injection 
  ""
  [ base_fun ]
  (py/call-attr vgg19 "keras_modules_injection"  base_fun ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr vgg19 "preprocess_input"  ))
