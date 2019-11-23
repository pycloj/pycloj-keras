(ns keras.initializers.identity
  "Initializer that generates the identity matrix.

    Only use for 2D matrices.
    If the long side of the matrix is a multiple of the short side,
    multiple identity matrices are concatenated along the long side.

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

(defn identity 
  "Initializer that generates the identity matrix.

    Only use for 2D matrices.
    If the long side of the matrix is a multiple of the short side,
    multiple identity matrices are concatenated along the long side.

    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    "
  [ & {:keys [gain]
       :or {gain 1.0}} ]
  
   (py/call-attr-kw initializers "identity" [] {:gain gain }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr initializers "get_config"  self ))
