(ns keras.utils.data-utils.HTTPError
  "Raised when HTTP error occurs, but also acts like non-error return"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-utils (import-module "keras.utils.data_utils"))

(defn HTTPError 
  "Raised when HTTP error occurs, but also acts like non-error return"
  [ & {:keys [url code msg hdrs fp]} ]
   (py/call-attr-kw data-utils "HTTPError" [] {:url url :code code :msg msg :hdrs hdrs :fp fp }))

(defn close 
  "
        Close the temporary file, possibly deleting it.
        "
  [ self ]
  (py/call-attr data-utils "close"  self ))

(defn getcode 
  ""
  [ self ]
  (py/call-attr data-utils "getcode"  self ))

(defn geturl 
  ""
  [ self ]
  (py/call-attr data-utils "geturl"  self ))

(defn headers 
  ""
  [ self ]
    (py/call-attr data-utils "headers"  self))

(defn info 
  ""
  [ self ]
  (py/call-attr data-utils "info"  self ))

(defn reason 
  ""
  [ self ]
    (py/call-attr data-utils "reason"  self))
