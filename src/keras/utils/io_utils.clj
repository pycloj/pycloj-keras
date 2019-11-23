(ns keras.utils.io-utils
  "Utilities related to disk I/O."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io-utils (import-module "keras.utils.io_utils"))

(defn ask-to-proceed-with-overwrite 
  "Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    "
  [ & {:keys [filepath]} ]
   (py/call-attr-kw io-utils "ask_to_proceed_with_overwrite" [] {:filepath filepath }))
