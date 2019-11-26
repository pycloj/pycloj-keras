(ns keras.constraints
  "Constraints: functions that impose constraints on weight values.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "keras.constraints"))
(defn deserialize 
  ""
  [config  & {:keys [custom_objects]} ]
    (py/call-attr-kw constraints "deserialize" [config] {:custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw constraints "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  ""
  [ identifier ]
  (py/call-attr constraints "get"  identifier ))

(defn serialize 
  ""
  [ constraint ]
  (py/call-attr constraints "serialize"  constraint ))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr constraints "serialize_keras_object"  instance ))
