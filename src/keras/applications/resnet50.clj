(ns keras.applications.resnet50
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resnet50 (import-module "keras.applications.resnet50"))

(defn ResNet50 
  ""
  [  ]
  (py/call-attr resnet50 "ResNet50"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr resnet50 "decode_predictions"  ))

(defn keras-modules-injection 
  ""
  [ base_fun ]
  (py/call-attr resnet50 "keras_modules_injection"  base_fun ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr resnet50 "preprocess_input"  ))
