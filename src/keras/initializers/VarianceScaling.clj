(ns keras.initializers.VarianceScaling
  "Initializer capable of adapting its scale to the shape of weights.

    With `distribution=\"normal\"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

        - number of input units in the weight tensor, if mode = \"fan_in\"
        - number of output units, if mode = \"fan_out\"
        - average of the numbers of input and output units, if mode = \"fan_avg\"

    With `distribution=\"uniform\"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    # Arguments
        scale: Scaling factor (positive float).
        mode: One of \"fan_in\", \"fan_out\", \"fan_avg\".
        distribution: Random distribution to use. One of \"normal\", \"uniform\".
        seed: A Python integer. Used to seed the random generator.

    # Raises
        ValueError: In case of an invalid value for the \"scale\", mode\" or
          \"distribution\" arguments.
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

(defn VarianceScaling 
  "Initializer capable of adapting its scale to the shape of weights.

    With `distribution=\"normal\"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

        - number of input units in the weight tensor, if mode = \"fan_in\"
        - number of output units, if mode = \"fan_out\"
        - average of the numbers of input and output units, if mode = \"fan_avg\"

    With `distribution=\"uniform\"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    # Arguments
        scale: Scaling factor (positive float).
        mode: One of \"fan_in\", \"fan_out\", \"fan_avg\".
        distribution: Random distribution to use. One of \"normal\", \"uniform\".
        seed: A Python integer. Used to seed the random generator.

    # Raises
        ValueError: In case of an invalid value for the \"scale\", mode\" or
          \"distribution\" arguments.
    "
  [ & {:keys [scale mode distribution seed]
       :or {scale 1.0 mode "fan_in" distribution "normal"}} ]
  
   (py/call-attr-kw initializers "VarianceScaling" [] {:scale scale :mode mode :distribution distribution :seed seed }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr initializers "get_config"  self ))
