(ns keras.applications.mobilenet-v2
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mobilenet-v2 (import-module "keras.applications.mobilenet_v2"))

(defn MobileNetV2 
  ""
  [  ]
  (py/call-attr mobilenet-v2 "MobileNetV2"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr mobilenet-v2 "decode_predictions"  ))

(defn keras-modules-injection 
  ""
  [ base_fun ]
  (py/call-attr mobilenet-v2 "keras_modules_injection"  base_fun ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr mobilenet-v2 "preprocess_input"  ))
