(ns keras.engine.saving
  "Model saving utilities.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce saving (import-module "keras.engine.saving"))

(defn allow-read-from-gcs 
  "Function decorator to support loading from Google Cloud Storage (GCS).

    This decorator parses the `filepath` argument of the `load_function` and
    fetches the required object from GCS if `filepath` starts with \"gs://\".

    Note: the file is temporarily copied to local filesystem from GCS before loaded.

    # Arguments
        load_function: The function to wrap, with requirements:
            - should have one _named_ argument `filepath` indicating the location to
            load from.
    "
  [ load_function ]
  (py/call-attr saving "allow_read_from_gcs"  load_function ))

(defn allow-write-to-gcs 
  "Function decorator to support saving to Google Cloud Storage (GCS).

    This decorator parses the `filepath` argument of the `save_function` and
    transfers the file to GCS if `filepath` starts with \"gs://\".

    Note: the file is temporarily writen to local filesystem before copied to GSC.

    # Arguments
        save_function: The function to wrap, with requirements:
            - second positional argument should indicate the location to save to.
            - third positional argument should be the `overwrite` option indicating
            whether we should overwrite an existing file/object at the target
            location, or instead ask the user with a manual prompt.
    "
  [ save_function ]
  (py/call-attr saving "allow_write_to_gcs"  save_function ))

(defn ask-to-proceed-with-overwrite 
  "Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    "
  [ filepath ]
  (py/call-attr saving "ask_to_proceed_with_overwrite"  filepath ))

(defn getargspec 
  "Get the names and default values of a callable object's parameters.

    A tuple of seven things is returned:
    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations).
    'args' is a list of the parameter names.
    'varargs' and 'varkw' are the names of the * and ** parameters or None.
    'defaults' is an n-tuple of the default values of the last n parameters.
    'kwonlyargs' is a list of keyword-only parameter names.
    'kwonlydefaults' is a dictionary mapping names from kwonlyargs to defaults.
    'annotations' is a dictionary mapping parameter names to annotations.

    Notable differences from inspect.signature():
      - the \"self\" parameter is always reported, even for bound methods
      - wrapper chains defined by __wrapped__ *not* unwrapped automatically
    "
  [ func ]
  (py/call-attr saving "getargspec"  func ))

(defn load-attributes-from-hdf5-group 
  "Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    # Returns
        data: Attributes data.
    "
  [ group name ]
  (py/call-attr saving "load_attributes_from_hdf5_group"  group name ))

(defn load-from-binary-h5py 
  "Calls `load_function` on a `h5py.File` read from the binary `stream`.

    # Arguments
        load_function: A function that takes a `h5py.File`, reads from it, and
            returns any object.
        stream: Any file-like object implementing the method `read` that returns
            `bytes` data (e.g. `io.BytesIO`) that represents a valid h5py file image.

    # Returns
        The object returned by `load_function`.
    "
  [ load_function stream ]
  (py/call-attr saving "load_from_binary_h5py"  load_function stream ))

(defn load-model 
  "Loads a model saved via `save_model`.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model
            - h5py.File or h5py.Group object from which to load the model
            - any file-like object implementing the method `read` that returns
            `bytes` data (e.g. `io.BytesIO`) that represents a valid h5py file image.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    "
  [filepath & {:keys [custom_objects compile]
                       :or {compile true}} ]
    (py/call-attr-kw saving "load_model" [filepath] {:custom_objects custom_objects :compile compile }))

(defn load-weights-from-hdf5-group 
  "Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    "
  [f layers & {:keys [reshape]
                       :or {reshape false}} ]
    (py/call-attr-kw saving "load_weights_from_hdf5_group" [f layers] {:reshape reshape }))

(defn load-weights-from-hdf5-group-by-name 
  "Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    "
  [f layers & {:keys [skip_mismatch reshape]
                       :or {skip_mismatch false reshape false}} ]
    (py/call-attr-kw saving "load_weights_from_hdf5_group_by_name" [f layers] {:skip_mismatch skip_mismatch :reshape reshape }))
(defn model-from-config 
  "Instantiates a Keras model from its config.

    # Arguments
        config: Configuration dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    # Returns
        A Keras model instance (uncompiled).

    # Raises
        TypeError: if `config` is not a dictionary.
    "
  [config  & {:keys [custom_objects]} ]
    (py/call-attr-kw saving "model_from_config" [config] {:custom_objects custom_objects }))
(defn model-from-json 
  "Parses a JSON model configuration file and returns a model instance.

    # Arguments
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    # Returns
        A Keras model instance (uncompiled).
    "
  [json_string  & {:keys [custom_objects]} ]
    (py/call-attr-kw saving "model_from_json" [json_string] {:custom_objects custom_objects }))
(defn model-from-yaml 
  "Parses a yaml model configuration file and returns a model instance.

    # Arguments
        yaml_string: YAML string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    # Returns
        A Keras model instance (uncompiled).
    "
  [yaml_string  & {:keys [custom_objects]} ]
    (py/call-attr-kw saving "model_from_yaml" [yaml_string] {:custom_objects custom_objects }))

(defn pickle-model 
  ""
  [ model ]
  (py/call-attr saving "pickle_model"  model ))

(defn preprocess-weights-for-loading 
  "Converts layers weights from Keras 1 format to Keras 2.

    # Arguments
        layer: Layer instance.
        weights: List of weights values (Numpy arrays).
        original_keras_version: Keras version for the weights, as a string.
        original_backend: Keras backend the weights were trained with,
            as a string.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Returns
        A list of weights values (Numpy arrays).
    "
  [layer weights & {:keys [original_keras_version original_backend reshape]
                       :or {reshape false}} ]
    (py/call-attr-kw saving "preprocess_weights_for_loading" [layer weights] {:original_keras_version original_keras_version :original_backend original_backend :reshape reshape }))

(defn save-attributes-to-hdf5-group 
  "Saves attributes (data) of the specified name into the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not
    able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to save.
        data: Attributes data to store.
    "
  [ group name data ]
  (py/call-attr saving "save_attributes_to_hdf5_group"  group name data ))

(defn save-model 
  "Save a model to a HDF5 file.

    Note: Please also see
    [How can I install HDF5 or h5py to save my models in Keras?](
        /getting-started/faq/
        #how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)
    in the FAQ for instructions on how to install `h5py`.

    The saved model contains:
        - the model's configuration (topology)
        - the model's weights
        - the model's optimizer's state (if any)

    Thus the saved model can be reinstantiated in
    the exact same state, without any of the code
    used for model definition or training.

    # Arguments
        model: Keras model instance to be saved.
        filepath: one of the following:
            - string, path to the file to save the model to
            - h5py.File or h5py.Group object where to save the model
            - any file-like object implementing the method `write` that accepts
                `bytes` data (e.g. `io.BytesIO`).
        overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.

    # Raises
        ImportError: if h5py is not available.
    "
  [model filepath & {:keys [overwrite include_optimizer]
                       :or {overwrite true include_optimizer true}} ]
    (py/call-attr-kw saving "save_model" [model filepath] {:overwrite overwrite :include_optimizer include_optimizer }))

(defn save-to-binary-h5py 
  "Calls `save_function` on an in memory `h5py.File`.

    The file is subsequently written to the binary `stream`.

     # Arguments
        save_function: A function that takes a `h5py.File`, writes to it and
            (optionally) returns any object.
        stream: Any file-like object implementing the method `write` that accepts
            `bytes` data (e.g. `io.BytesIO`).
     "
  [ save_function stream ]
  (py/call-attr saving "save_to_binary_h5py"  save_function stream ))

(defn save-weights-to-hdf5-group 
  "Saves weights into the HDF5 group.

    # Arguments
        group: A pointer to a HDF5 group.
        layers: Layers to load.
    "
  [ group layers ]
  (py/call-attr saving "save_weights_to_hdf5_group"  group layers ))

(defn unpickle-model 
  ""
  [ state ]
  (py/call-attr saving "unpickle_model"  state ))

(defn wraps 
  "Decorator factory to apply update_wrapper() to a wrapper function

       Returns a decorator that invokes update_wrapper() with the decorated
       function as the wrapper argument and the arguments to wraps() as the
       remaining arguments. Default arguments are as for update_wrapper().
       This is a convenience function to simplify applying partial() to
       update_wrapper().
    "
  [wrapped & {:keys [assigned updated]
                       :or {assigned ('__module__', '__name__', '__qualname__', '__doc__', '__annotations__') updated ('__dict__',)}} ]
    (py/call-attr-kw saving "wraps" [wrapped] {:assigned assigned :updated updated }))
