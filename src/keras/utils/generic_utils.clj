(ns keras.utils.generic-utils
  "Python utilities required by Keras."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce generic-utils (import-module "keras.utils.generic_utils"))

(defn check-for-unexpected-keys 
  ""
  [ name input_dict expected_values ]
  (py/call-attr generic-utils "check_for_unexpected_keys"  name input_dict expected_values ))

(defn custom-object-scope 
  "Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

    Convenience wrapper for `CustomObjectScope`.
    Code within a `with` statement will be able to access custom objects
    by name. Changes to global custom objects persist
    within the enclosing `with` statement. At end of the `with` statement,
    global custom objects are reverted to state
    at beginning of the `with` statement.

    # Example

    Consider a custom object `MyObject`

    ```python
        with custom_object_scope({'MyObject':MyObject}):
            layer = Dense(..., kernel_regularizer='MyObject')
            # save, load, etc. will recognize custom object by name
    ```

    # Arguments
        *args: Variable length list of dictionaries of name,
            class pairs to add to custom objects.

    # Returns
        Object of type `CustomObjectScope`.
    "
  [  ]
  (py/call-attr generic-utils "custom_object_scope"  ))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw generic-utils "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn func-dump 
  "Serializes a user defined function.

    # Arguments
        func: the function to serialize.

    # Returns
        A tuple `(code, defaults, closure)`.
    "
  [ func ]
  (py/call-attr generic-utils "func_dump"  func ))
(defn func-load 
  "Deserializes a user defined function.

    # Arguments
        code: bytecode of the function.
        defaults: defaults of the function.
        closure: closure of the function.
        globs: dictionary of global objects.

    # Returns
        A function object.
    "
  [code  & {:keys [defaults closure globs]} ]
    (py/call-attr-kw generic-utils "func_load" [code] {:defaults defaults :closure closure :globs globs }))

(defn get-custom-objects 
  "Retrieves a live reference to the global dictionary of custom objects.

    Updating and clearing custom objects using `custom_object_scope`
    is preferred, but `get_custom_objects` can
    be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

    # Example

    ```python
        get_custom_objects().clear()
        get_custom_objects()['MyObject'] = MyObject
    ```

    # Returns
        Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
    "
  [  ]
  (py/call-attr generic-utils "get_custom_objects"  ))

(defn getargspec 
  "Python 2/3 compatible `getargspec`.

    Calls `getfullargspec` and assigns args, varargs,
    varkw, and defaults to a python 2/3 compatible `ArgSpec`.
    The parameter name 'varkw' is changed to 'keywords' to fit the
    `ArgSpec` struct.

    # Arguments
        fn: the target function to inspect.

    # Returns
        An ArgSpec with args, varargs, keywords, and defaults parameters
        from FullArgSpec.
    "
  [ fn ]
  (py/call-attr generic-utils "getargspec"  fn ))

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
  [fn name & {:keys [accept_all]
                       :or {accept_all false}} ]
    (py/call-attr-kw generic-utils "has_arg" [fn name] {:accept_all accept_all }))

(defn is-all-none 
  ""
  [ iterable_or_element ]
  (py/call-attr generic-utils "is_all_none"  iterable_or_element ))

(defn object-list-uid 
  ""
  [ object_list ]
  (py/call-attr generic-utils "object_list_uid"  object_list ))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr generic-utils "serialize_keras_object"  instance ))
(defn slice-arrays 
  "Slices an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `_slice_arrays(x, indices)`

    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    # Returns
        A slice of the array(s).
    "
  [arrays  & {:keys [start stop]} ]
    (py/call-attr-kw generic-utils "slice_arrays" [arrays] {:start start :stop stop }))

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
  [x & {:keys [allow_tuple]
                       :or {allow_tuple false}} ]
    (py/call-attr-kw generic-utils "to_list" [x] {:allow_tuple allow_tuple }))

(defn transpose-shape 
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
  [ shape target_format spatial_axes ]
  (py/call-attr generic-utils "transpose_shape"  shape target_format spatial_axes ))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument
        x: A list or tuple.

    # Returns
        The same iterable or the first element.
    "
  [ x ]
  (py/call-attr generic-utils "unpack_singleton"  x ))
