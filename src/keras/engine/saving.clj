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

(defn ask-to-proceed-with-overwrite 
  "Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    "
  [ & {:keys [filepath]} ]
   (py/call-attr-kw saving "ask_to_proceed_with_overwrite" [] {:filepath filepath }))

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
  [ & {:keys [group name]} ]
   (py/call-attr-kw saving "load_attributes_from_hdf5_group" [] {:group group :name name }))

(defn load-model 
  "Loads a model saved via `save_model`.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File or h5py.Group object from which to load the model
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
  [ & {:keys [filepath custom_objects compile]
       :or {compile true}} ]
  
   (py/call-attr-kw saving "load_model" [] {:filepath filepath :custom_objects custom_objects :compile compile }))

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
  [ & {:keys [f layers reshape]
       :or {reshape false}} ]
  
   (py/call-attr-kw saving "load_weights_from_hdf5_group" [] {:f f :layers layers :reshape reshape }))

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
  [ & {:keys [f layers skip_mismatch reshape]
       :or {skip_mismatch false reshape false}} ]
  
   (py/call-attr-kw saving "load_weights_from_hdf5_group_by_name" [] {:f f :layers layers :skip_mismatch skip_mismatch :reshape reshape }))

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
  [ & {:keys [config custom_objects]} ]
   (py/call-attr-kw saving "model_from_config" [] {:config config :custom_objects custom_objects }))

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
  [ & {:keys [json_string custom_objects]} ]
   (py/call-attr-kw saving "model_from_json" [] {:json_string json_string :custom_objects custom_objects }))

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
  [ & {:keys [yaml_string custom_objects]} ]
   (py/call-attr-kw saving "model_from_yaml" [] {:yaml_string yaml_string :custom_objects custom_objects }))

(defn pickle-model 
  ""
  [ & {:keys [model]} ]
   (py/call-attr-kw saving "pickle_model" [] {:model model }))

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
  [ & {:keys [layer weights original_keras_version original_backend reshape]
       :or {reshape false}} ]
  
   (py/call-attr-kw saving "preprocess_weights_for_loading" [] {:layer layer :weights weights :original_keras_version original_keras_version :original_backend original_backend :reshape reshape }))

(defn save-attributes-to-hdf5-group 
  "Saves attributes (data) of the specified name into the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not
    able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to save.
        data: Attributes data to store.
    "
  [ & {:keys [group name data]} ]
   (py/call-attr-kw saving "save_attributes_to_hdf5_group" [] {:group group :name name :data data }))

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
            - string, path where to save the model, or
            - h5py.File or h5py.Group object where to save the model
        overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.

    # Raises
        ImportError: if h5py is not available.
    "
  [ & {:keys [model filepath overwrite include_optimizer]
       :or {overwrite true include_optimizer true}} ]
  
   (py/call-attr-kw saving "save_model" [] {:model model :filepath filepath :overwrite overwrite :include_optimizer include_optimizer }))

(defn save-weights-to-hdf5-group 
  ""
  [ & {:keys [f layers]} ]
   (py/call-attr-kw saving "save_weights_to_hdf5_group" [] {:f f :layers layers }))

(defn unpickle-model 
  ""
  [ & {:keys [state]} ]
   (py/call-attr-kw saving "unpickle_model" [] {:state state }))
