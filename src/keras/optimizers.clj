(ns keras.optimizers
  "Built-in optimizer classes.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimizers (import-module "keras.optimizers"))

(defn clip-norm 
  "Clip the gradient `g` if the L2 norm `n` exceeds `c`.

    # Arguments
        g: Tensor, the gradient tensor
        c: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        n: Tensor, actual norm of `g`.

    # Returns
        Tensor, the gradient clipped if required.
    "
  [ g c n ]
  (py/call-attr optimizers "clip_norm"  g c n ))
(defn deserialize 
  "Inverse of the `serialize` function.

    # Arguments
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping
            names (strings) to custom objects
            (classes and functions)
            to be considered during deserialization.

    # Returns
        A Keras Optimizer instance.
    "
  [config  & {:keys [custom_objects]} ]
    (py/call-attr-kw optimizers "deserialize" [config] {:custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw optimizers "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  "Retrieves a Keras Optimizer instance.

    # Arguments
        identifier: Optimizer identifier, one of
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).
            - TensorFlow Optimizer instance
                (it will be wrapped as a Keras Optimizer).

    # Returns
        A Keras Optimizer instance.

    # Raises
        ValueError: If `identifier` cannot be interpreted.
    "
  [ identifier ]
  (py/call-attr optimizers "get"  identifier ))

(defn serialize 
  ""
  [ optimizer ]
  (py/call-attr optimizers "serialize"  optimizer ))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr optimizers "serialize_keras_object"  instance ))
