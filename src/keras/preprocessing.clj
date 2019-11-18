
(ns keras.preprocessing
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw
                     att-type-map
                     ->py-dict
                     ->py-list
                     ]
             :as py]
            [clojure.pprint :as pp]))

(py/initialize!)
(defonce keras.preprocessing (import-module "keras.preprocessing"))

(defn Input 
  "`Input()` is used to instantiate a Keras tensor.

    A Keras tensor is a tensor object from the underlying backend
    (Theano, TensorFlow or CNTK), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.

    For instance, if a, b and c are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    The added Keras attributes are:
        `_keras_shape`: Integer shape tuple propagated
            via Keras-side shape inference.
        `_keras_history`: Last layer applied to the tensor.
            the entire layer graph is retrievable from that layer,
            recursively.

    # Arguments
        shape: A shape tuple (integer), not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors.
        batch_shape: A shape tuple (integer), including the batch size.
            For instance, `batch_shape=(10, 32)` indicates that
            the expected input will be batches of 10 32-dimensional vectors.
            `batch_shape=(None, 32)` indicates batches of an arbitrary number
            of 32-dimensional vectors.
        name: An optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
        sparse: A boolean specifying whether the placeholder
            to be created is sparse.
        tensor: Optional existing tensor to wrap into the `Input` layer.
            If set, the layer will not create a placeholder tensor.

    # Returns
        A tensor.

    # Example

    ```python
    # this is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
    "
  [ & {:keys [shape batch_shape name dtype sparse tensor]
       :or {sparse false}} ]
  
   (py/call-attr-kw keras.preprocessing "Input" [] {:shape shape :batch_shape batch_shape :name name :dtype dtype :sparse sparse :tensor tensor }))
