(ns keras.applications.inception-resnet-v2
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce inception-resnet-v2 (import-module "keras.applications.inception_resnet_v2"))

(defn InceptionResNetV2 
  ""
  [  ]
  (py/call-attr inception-resnet-v2 "InceptionResNetV2"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr inception-resnet-v2 "decode_predictions"  ))

(defn keras-modules-injection 
  ""
  [ base_fun ]
  (py/call-attr inception-resnet-v2 "keras_modules_injection"  base_fun ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr inception-resnet-v2 "preprocess_input"  ))
