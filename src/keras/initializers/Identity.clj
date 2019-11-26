(ns keras.initializers.Identity
  "Initializer that generates the identity matrix.

    Only use for 2D matrices.
    If the desired matrix is not square, it gets padded
    with zeros for the additional rows/columns.

    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "keras.initializers"))

(defn Identity 
  "Initializer that generates the identity matrix.

    Only use for 2D matrices.
    If the desired matrix is not square, it gets padded
    with zeros for the additional rows/columns.

    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    "
  [ & {:keys [gain]
       :or {gain 1.0}} ]
  
   (py/call-attr-kw initializers "Identity" [] {:gain gain }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
