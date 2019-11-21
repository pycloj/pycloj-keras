
(ns keras.layers.layers
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
(defonce layers (import-module "keras.layers.layers"))

(defn AtrousConv1D [  ]
  ""
  (py/call-attr layers "AtrousConv1D"   ))

(defn AtrousConv2D [  ]
  ""
  (py/call-attr layers "AtrousConv2D"   ))

(defn AtrousConvolution1D [  ]
  ""
  (py/call-attr layers "AtrousConvolution1D"   ))

(defn AtrousConvolution2D [  ]
  ""
  (py/call-attr layers "AtrousConvolution2D"   ))

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
  
   (py/call-attr-kw layers "Input" [] {:shape shape :batch_shape batch_shape :name name :dtype dtype :sparse sparse :tensor tensor }))

(defn add [ & {:keys [inputs]} ]
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
   (py/call-attr-kw layers "add" [] {:inputs inputs }))

(defn average [ & {:keys [inputs]} ]
  "Functional interface to the `Average` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the average of the inputs.
    "
   (py/call-attr-kw layers "average" [] {:inputs inputs }))

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
  
   (py/call-attr-kw layers "concatenate" [] {:inputs inputs :axis axis }))

(defn deserialize [ & {:keys [config custom_objects]} ]
  "Instantiate a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)
    "
   (py/call-attr-kw layers "deserialize" [] {:config config :custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [ & {:keys [identifier module_objects custom_objects printable_module_name]
       :or {printable_module_name "object"}} ]
  
   (py/call-attr-kw layers "deserialize_keras_object" [] {:identifier identifier :module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

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
  
   (py/call-attr-kw layers "dot" [] {:inputs inputs :axes axes :normalize normalize }))

(defn func-dump [ & {:keys [func]} ]
  "Serializes a user defined function.

    # Arguments
        func: the function to serialize.

    # Returns
        A tuple `(code, defaults, closure)`.
    "
   (py/call-attr-kw layers "func_dump" [] {:func func }))

(defn func-load [ & {:keys [code defaults closure globs]} ]
  "Deserializes a user defined function.

    # Arguments
        code: bytecode of the function.
        defaults: defaults of the function.
        closure: closure of the function.
        globs: dictionary of global objects.

    # Returns
        A function object.
    "
   (py/call-attr-kw layers "func_load" [] {:code code :defaults defaults :closure closure :globs globs }))

(defn has-arg 
  "Checks if a callable accepts a given keyword argument.

    For Python 2, checks if there is an argument with the given name.

    For Python 3, checks if there is an argument with the given name, and
    also whether this argument can be called with a keyword (i.e. if it is
    not a positional-only argument).

    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name`
                    but the function accepts a `**kwargs` argument.

    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    "
  [ & {:keys [fn name accept_all]
       :or {accept_all false}} ]
  
   (py/call-attr-kw layers "has_arg" [] {:fn fn :name name :accept_all accept_all }))

(defn maximum [ & {:keys [inputs]} ]
  "Functional interface to the `Maximum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise maximum of the inputs.
    "
   (py/call-attr-kw layers "maximum" [] {:inputs inputs }))

(defn minimum [ & {:keys [inputs]} ]
  "Functional interface to the `Minimum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise minimum of the inputs.
    "
   (py/call-attr-kw layers "minimum" [] {:inputs inputs }))

(defn multiply [ & {:keys [inputs]} ]
  "Functional interface to the `Multiply` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise product of the inputs.
    "
   (py/call-attr-kw layers "multiply" [] {:inputs inputs }))

(defn namedtuple 
  "Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    "
  [ & {:keys [typename field_names rename defaults module]
       :or {rename false}} ]
  
   (py/call-attr-kw layers "namedtuple" [] {:typename typename :field_names field_names :rename rename :defaults defaults :module module }))

(defn object-list-uid [ & {:keys [object_list]} ]
  ""
   (py/call-attr-kw layers "object_list_uid" [] {:object_list object_list }))

(defn serialize [ & {:keys [layer]} ]
  "Serialize a layer.

    # Arguments
        layer: a Layer object.

    # Returns
        dictionary with config.
    "
   (py/call-attr-kw layers "serialize" [] {:layer layer }))

(defn subtract [ & {:keys [inputs]} ]
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
   (py/call-attr-kw layers "subtract" [] {:inputs inputs }))

(defn to-list 
  "Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    "
  [ & {:keys [x allow_tuple]
       :or {allow_tuple false}} ]
  
   (py/call-attr-kw layers "to_list" [] {:x x :allow_tuple allow_tuple }))

(defn transpose-shape [ & {:keys [shape target_format spatial_axes]} ]
  "Converts a tuple or a list to the correct `data_format`.

    It does so by switching the positions of its elements.

    # Arguments
        shape: Tuple or list, often representing shape,
            corresponding to `'channels_last'`.
        target_format: A string, either `'channels_first'` or `'channels_last'`.
        spatial_axes: A tuple of integers.
            Correspond to the indexes of the spatial axes.
            For example, if you pass a shape
            representing (batch_size, timesteps, rows, cols, channels),
            then `spatial_axes=(2, 3)`.

    # Returns
        A tuple or list, with the elements permuted according
        to `target_format`.

    # Example
    ```python
        >>> from keras.utils.generic_utils import transpose_shape
        >>> transpose_shape((16, 128, 128, 32),'channels_first', spatial_axes=(1, 2))
        (16, 32, 128, 128)
        >>> transpose_shape((16, 128, 128, 32), 'channels_last', spatial_axes=(1, 2))
        (16, 128, 128, 32)
        >>> transpose_shape((128, 128, 32), 'channels_first', spatial_axes=(0, 1))
        (32, 128, 128)
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    "
   (py/call-attr-kw layers "transpose_shape" [] {:shape shape :target_format target_format :spatial_axes spatial_axes }))
