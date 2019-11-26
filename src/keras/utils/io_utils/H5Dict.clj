(ns keras.utils.io-utils.H5Dict
  " A dict-like wrapper around h5py groups (or dicts).

    This allows us to have a single serialization logic
    for both pickling and saving to disk.

    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization/
    deserialization.

    # Arguments
        path: Either a string (path on disk), a Path, a dict, or a HDF5 Group.
        mode: File open mode (one of `{\"a\", \"r\", \"w\"}`).
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io-utils (import-module "keras.utils.io_utils"))

(defn H5Dict 
  " A dict-like wrapper around h5py groups (or dicts).

    This allows us to have a single serialization logic
    for both pickling and saving to disk.

    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization/
    deserialization.

    # Arguments
        path: Either a string (path on disk), a Path, a dict, or a HDF5 Group.
        mode: File open mode (one of `{\"a\", \"r\", \"w\"}`).
    "
  [path & {:keys [mode]
                       :or {mode "a"}} ]
    (py/call-attr-kw io-utils "H5Dict" [path] {:mode mode }))

(defn close 
  ""
  [ self  ]
  (py/call-attr self "close"  self  ))
(defn get 
  ""
  [self key  & {:keys [default]} ]
    (py/call-attr-kw self "get" [key] {:default default }))

(defn is-supported-type 
  "Check if `path` is of supported type for instantiating a `H5Dict`"
  [ self path ]
  (py/call-attr self "is_supported_type"  self path ))

(defn iter 
  ""
  [ self  ]
  (py/call-attr self "iter"  self  ))

(defn update 
  ""
  [ self  ]
  (py/call-attr self "update"  self  ))
