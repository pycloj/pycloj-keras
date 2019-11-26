(ns keras.datasets.cifar
  "Utilities common to CIFAR10 and CIFAR100 datasets.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cifar (import-module "keras.datasets.cifar"))

(defn load-batch 
  "Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    "
  [fpath & {:keys [label_key]
                       :or {label_key "labels"}} ]
    (py/call-attr-kw cifar "load_batch" [fpath] {:label_key label_key }))
