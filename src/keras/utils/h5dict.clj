(ns keras.utils.h5dict
  " A dict-like wrapper around h5py groups (or dicts).

    This allows us to have a single serialization logic
    for both pickling and saving to disk.

    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization/
    deserialization.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "keras.utils"))

(defn h5dict 
  " A dict-like wrapper around h5py groups (or dicts).

    This allows us to have a single serialization logic
    for both pickling and saving to disk.

    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization/
    deserialization.
    "
  [ & {:keys [path mode]
       :or {mode "a"}} ]
  
   (py/call-attr-kw utils "h5dict" [] {:path path :mode mode }))

(defn close 
  ""
  [ self ]
  (py/call-attr utils "close"  self ))

(defn get 
  ""
  [self  & {:keys [key default]} ]
    (py/call-attr-kw utils "get" [self] {:key key :default default }))

(defn iter 
  ""
  [ self ]
  (py/call-attr utils "iter"  self ))

(defn update 
  ""
  [ self ]
  (py/call-attr utils "update"  self ))
