(ns keras.callbacks.Iterable
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce callbacks (import-module "keras.callbacks"))

(defn Iterable 
  ""
  [  ]
  (py/call-attr callbacks "Iterable"   ))
