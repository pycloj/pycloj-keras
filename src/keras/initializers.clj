(ns keras.initializers
  "Built-in weight initializers.
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

(defn deserialize 
  ""
  [ & {:keys [config custom_objects]} ]
   (py/call-attr-kw initializers "deserialize" [] {:config config :custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [ & {:keys [identifier module_objects custom_objects printable_module_name]
       :or {printable_module_name "object"}} ]
  
   (py/call-attr-kw initializers "deserialize_keras_object" [] {:identifier identifier :module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  ""
  [ & {:keys [identifier]} ]
   (py/call-attr-kw initializers "get" [] {:identifier identifier }))

(defn glorot-normal 
  "Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    "
  [ & {:keys [seed]} ]
   (py/call-attr-kw initializers "glorot_normal" [] {:seed seed }))

(defn glorot-uniform 
  "Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    "
  [ & {:keys [seed]} ]
   (py/call-attr-kw initializers "glorot_uniform" [] {:seed seed }))

(defn he-normal 
  "He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        He et al., http://arxiv.org/abs/1502.01852
    "
  [ & {:keys [seed]} ]
   (py/call-attr-kw initializers "he_normal" [] {:seed seed }))

(defn he-uniform 
  "He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        He et al., http://arxiv.org/abs/1502.01852
    "
  [ & {:keys [seed]} ]
   (py/call-attr-kw initializers "he_uniform" [] {:seed seed }))

(defn lecun-normal 
  "LeCun normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
        - [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    "
  [ & {:keys [seed]} ]
   (py/call-attr-kw initializers "lecun_normal" [] {:seed seed }))

(defn lecun-uniform 
  "LeCun uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        LeCun 98, Efficient Backprop,
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    "
  [ & {:keys [seed]} ]
   (py/call-attr-kw initializers "lecun_uniform" [] {:seed seed }))

(defn serialize 
  ""
  [ & {:keys [initializer]} ]
   (py/call-attr-kw initializers "serialize" [] {:initializer initializer }))

(defn serialize-keras-object 
  ""
  [ & {:keys [instance]} ]
   (py/call-attr-kw initializers "serialize_keras_object" [] {:instance instance }))
