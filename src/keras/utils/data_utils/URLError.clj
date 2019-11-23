(ns keras.utils.data-utils.URLError
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-utils (import-module "keras.utils.data_utils"))

(defn URLError 
  ""
  [ & {:keys [reason filename]} ]
   (py/call-attr-kw data-utils "URLError" [] {:reason reason :filename filename }))
