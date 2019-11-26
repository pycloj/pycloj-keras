(ns keras.regularizers
  "Built-in regularizers.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce regularizers (import-module "keras.regularizers"))
(defn deserialize 
  ""
  [config  & {:keys [custom_objects]} ]
    (py/call-attr-kw regularizers "deserialize" [config] {:custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw regularizers "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  ""
  [ identifier ]
  (py/call-attr regularizers "get"  identifier ))

(defn l1 
  ""
  [ & {:keys [l]
       :or {l 0.01}} ]
  
   (py/call-attr-kw regularizers "l1" [] {:l l }))

(defn l1-l2 
  ""
  [ & {:keys [l1 l2]
       :or {l1 0.01 l2 0.01}} ]
  
   (py/call-attr-kw regularizers "l1_l2" [] {:l1 l1 :l2 l2 }))

(defn l2 
  ""
  [ & {:keys [l]
       :or {l 0.01}} ]
  
   (py/call-attr-kw regularizers "l2" [] {:l l }))

(defn serialize 
  ""
  [ regularizer ]
  (py/call-attr regularizers "serialize"  regularizer ))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr regularizers "serialize_keras_object"  instance ))
