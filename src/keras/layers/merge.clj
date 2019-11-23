(ns keras.layers.merge
  "Layers that can merge several inputs into one.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce merge (import-module "keras.layers.merge"))

(defn add 
  "Functional interface to the `Add` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the sum of the inputs.

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        added = keras.layers.add([x1, x2])

        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    "
  [ & {:keys [inputs]} ]
   (py/call-attr-kw merge "add" [] {:inputs inputs }))

(defn average 
  "Functional interface to the `Average` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the average of the inputs.
    "
  [ & {:keys [inputs]} ]
   (py/call-attr-kw merge "average" [] {:inputs inputs }))

(defn concatenate 
  "Functional interface to the `Concatenate` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the concatenation of the inputs alongside axis `axis`.
    "
  [ & {:keys [inputs axis]
       :or {axis -1}} ]
  
   (py/call-attr-kw merge "concatenate" [] {:inputs inputs :axis axis }))

(defn dot 
  "Functional interface to the `Dot` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the dot product of the samples from the inputs.
    "
  [ & {:keys [inputs axes normalize]
       :or {normalize false}} ]
  
   (py/call-attr-kw merge "dot" [] {:inputs inputs :axes axes :normalize normalize }))

(defn maximum 
  "Functional interface to the `Maximum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise maximum of the inputs.
    "
  [ & {:keys [inputs]} ]
   (py/call-attr-kw merge "maximum" [] {:inputs inputs }))

(defn minimum 
  "Functional interface to the `Minimum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise minimum of the inputs.
    "
  [ & {:keys [inputs]} ]
   (py/call-attr-kw merge "minimum" [] {:inputs inputs }))

(defn multiply 
  "Functional interface to the `Multiply` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise product of the inputs.
    "
  [ & {:keys [inputs]} ]
   (py/call-attr-kw merge "multiply" [] {:inputs inputs }))

(defn subtract 
  "Functional interface to the `Subtract` layer.

    # Arguments
        inputs: A list of input tensors (exactly 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the difference of the inputs.

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        subtracted = keras.layers.subtract([x1, x2])

        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    "
  [ & {:keys [inputs]} ]
   (py/call-attr-kw merge "subtract" [] {:inputs inputs }))
