(ns keras.initializers.Orthogonal
  "Initializer that generates a random orthogonal matrix.

    # Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.

    # References
        - [Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks](http://arxiv.org/abs/1312.6120)
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

(defn Orthogonal 
  "Initializer that generates a random orthogonal matrix.

    # Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.

    # References
        - [Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks](http://arxiv.org/abs/1312.6120)
    "
  [ & {:keys [gain seed]
       :or {gain 1.0}} ]
  
   (py/call-attr-kw initializers "Orthogonal" [] {:gain gain :seed seed }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
