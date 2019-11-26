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
  [ filepath ]
  (py/call-attr io-utils "ask_to_proceed_with_overwrite"  filepath ))

(defn load-from-binary-h5py 
  "Calls `load_function` on a `h5py.File` read from the binary `stream`.

    # Arguments
        load_function: A function that takes a `h5py.File`, reads from it, and
            returns any object.
        stream: Any file-like object implementing the method `read` that returns
            `bytes` data (e.g. `io.BytesIO`) that represents a valid h5py file image.

    # Returns
        The object returned by `load_function`.
    "
  [ load_function stream ]
  (py/call-attr io-utils "load_from_binary_h5py"  load_function stream ))

(defn save-to-binary-h5py 
  "Calls `save_function` on an in memory `h5py.File`.

    The file is subsequently written to the binary `stream`.

     # Arguments
        save_function: A function that takes a `h5py.File`, writes to it and
            (optionally) returns any object.
        stream: Any file-like object implementing the method `write` that accepts
            `bytes` data (e.g. `io.BytesIO`).
     "
  [ save_function stream ]
  (py/call-attr io-utils "save_to_binary_h5py"  save_function stream ))
