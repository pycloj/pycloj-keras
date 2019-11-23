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
  [ & {:keys [config custom_objects]} ]
   (py/call-attr-kw constraints "deserialize" [] {:config config :custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [ & {:keys [identifier module_objects custom_objects printable_module_name]
       :or {printable_module_name "object"}} ]
  
   (py/call-attr-kw constraints "deserialize_keras_object" [] {:identifier identifier :module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  ""
  [ & {:keys [identifier]} ]
   (py/call-attr-kw constraints "get" [] {:identifier identifier }))

(defn serialize 
  ""
  [ & {:keys [constraint]} ]
   (py/call-attr-kw constraints "serialize" [] {:constraint constraint }))

(defn serialize-keras-object 
  ""
  [ & {:keys [instance]} ]
   (py/call-attr-kw constraints "serialize_keras_object" [] {:instance instance }))
