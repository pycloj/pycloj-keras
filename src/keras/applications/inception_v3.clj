(ns keras.applications.inception-v3
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce inception-v3 (import-module "keras.applications.inception_v3"))

(defn InceptionV3 
  ""
  [  ]
  (py/call-attr inception-v3 "InceptionV3"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr inception-v3 "decode_predictions"  ))

(defn keras-modules-injection 
  ""
  [ base_fun ]
  (py/call-attr inception-v3 "keras_modules_injection"  base_fun ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr inception-v3 "preprocess_input"  ))
