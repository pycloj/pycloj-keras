(ns keras.applications
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce applications (import-module "keras.applications"))

(defn DenseNet121 
  ""
  [  ]
  (py/call-attr applications "DenseNet121"   ))

(defn DenseNet169 
  ""
  [  ]
  (py/call-attr applications "DenseNet169"   ))

(defn DenseNet201 
  ""
  [  ]
  (py/call-attr applications "DenseNet201"   ))

(defn InceptionResNetV2 
  ""
  [  ]
  (py/call-attr applications "InceptionResNetV2"   ))

(defn InceptionV3 
  ""
  [  ]
  (py/call-attr applications "InceptionV3"   ))

(defn MobileNet 
  ""
  [  ]
  (py/call-attr applications "MobileNet"   ))

(defn MobileNetV2 
  ""
  [  ]
  (py/call-attr applications "MobileNetV2"   ))

(defn NASNetLarge 
  ""
  [  ]
  (py/call-attr applications "NASNetLarge"   ))

(defn NASNetMobile 
  ""
  [  ]
  (py/call-attr applications "NASNetMobile"   ))

(defn ResNet50 
  ""
  [  ]
  (py/call-attr applications "ResNet50"   ))

(defn VGG16 
  ""
  [  ]
  (py/call-attr applications "VGG16"   ))

(defn VGG19 
  ""
  [  ]
  (py/call-attr applications "VGG19"   ))

(defn Xception 
  ""
  [  ]
  (py/call-attr applications "Xception"   ))

(defn keras-modules-injection 
  ""
  [ & {:keys [base_fun]} ]
   (py/call-attr-kw applications "keras_modules_injection" [] {:base_fun base_fun }))
