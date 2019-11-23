(ns keras.utils
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "keras.utils"))

(defn convert-all-kernels-in-model 
  "Converts all convolution kernels in a model from Theano to TensorFlow.

    Also works from TensorFlow to Theano.

    # Arguments
        model: target model for the conversion.
    "
  [ & {:keys [model]} ]
   (py/call-attr-kw utils "convert_all_kernels_in_model" [] {:model model }))

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
  (py/call-attr utils "custom_object_scope"   ))

(defn deserialize-keras-object 
  ""
  [ & {:keys [identifier module_objects custom_objects printable_module_name]
       :or {printable_module_name "object"}} ]
  
   (py/call-attr-kw utils "deserialize_keras_object" [] {:identifier identifier :module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

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
  (py/call-attr utils "get_custom_objects"   ))

(defn get-file 
  "Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

    # Returns
        Path to the downloaded file
    "
  [ & {:keys [fname origin untar md5_hash file_hash cache_subdir hash_algorithm extract archive_format cache_dir]
       :or {untar false cache_subdir "datasets" hash_algorithm "auto" extract false archive_format "auto"}} ]
  
   (py/call-attr-kw utils "get_file" [] {:fname fname :origin origin :untar untar :md5_hash md5_hash :file_hash file_hash :cache_subdir cache_subdir :hash_algorithm hash_algorithm :extract extract :archive_format archive_format :cache_dir cache_dir }))

(defn get-source-inputs 
  "Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    # Arguments
        tensor: The tensor to start from.
        layer: Origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: Origin node index of the tensor.

    # Returns
        List of input tensors.
    "
  [ & {:keys [tensor layer node_index]} ]
   (py/call-attr-kw utils "get_source_inputs" [] {:tensor tensor :layer layer :node_index node_index }))

(defn multi-gpu-model 
  "Replicates a model on different GPUs.

    Specifically, this function implements single-machine
    multi-GPU data parallelism. It works in the following way:

    - Divide the model's input(s) into multiple sub-batches.
    - Apply a model copy on each sub-batch. Every model copy
        is executed on a dedicated GPU.
    - Concatenate the results (on CPU) into one big batch.

    E.g. if your `batch_size` is 64 and you use `gpus=2`,
    then we will divide the input into 2 sub-batches of 32 samples,
    process each sub-batch on one GPU, then return the full
    batch of 64 processed samples.

    This induces quasi-linear speedup on up to 8 GPUs.

    This function is only available with the TensorFlow backend
    for the time being.

    # Arguments
        model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or
            list of GPU IDs on which to create model replicas.
        cpu_merge: A boolean value to identify whether to force
            merging model weights under the scope of the CPU or not.
        cpu_relocation: A boolean value to identify whether to
            create the model's weights under the scope of the CPU.
            If the model is not defined under any preceding device
            scope, you can still rescue it by activating this option.

    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.

    # Example 1 - Training models with weights merge on CPU

    ```python
        import tensorflow as tf
        from keras.applications import Xception
        from keras.utils import multi_gpu_model
        import numpy as np

        num_samples = 1000
        height = 224
        width = 224
        num_classes = 1000

        # Instantiate the base model (or \"template\" model).
        # We recommend doing this with under a CPU device scope,
        # so that the model's weights are hosted on CPU memory.
        # Otherwise they may end up hosted on a GPU, which would
        # complicate weight sharing.
        with tf.device('/cpu:0'):
            model = Xception(weights=None,
                             input_shape=(height, width, 3),
                             classes=num_classes)

        # Replicates the model on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        parallel_model = multi_gpu_model(model, gpus=8)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')

        # Generate dummy data.
        x = np.random.random((num_samples, height, width, 3))
        y = np.random.random((num_samples, num_classes))

        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit(x, y, epochs=20, batch_size=256)

        # Save model via the template model (which shares the same weights):
        model.save('my_model.h5')
    ```

    # Example 2 - Training models with weights merge on CPU using cpu_relocation

    ```python
         ..
         # Not needed to change the device scope for model definition:
         model = Xception(weights=None, ..)

         try:
             model = multi_gpu_model(model, cpu_relocation=True)
             print(\"Training using multiple GPUs..\")
         except:
             print(\"Training using single GPU or CPU..\")

         model.compile(..)
         ..
    ```

    # Example 3 - Training models with weights merge on GPU (recommended for NV-link)

    ```python
         ..
         # Not needed to change the device scope for model definition:
         model = Xception(weights=None, ..)

         try:
             model = multi_gpu_model(model, cpu_merge=False)
             print(\"Training using multiple GPUs..\")
         except:
             print(\"Training using single GPU or CPU..\")

         model.compile(..)
         ..
    ```

    # On model saving

    To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
    with the template model (the argument you passed to `multi_gpu_model`),
    rather than the model returned by `multi_gpu_model`.
    "
  [ & {:keys [model gpus cpu_merge cpu_relocation]
       :or {cpu_merge true cpu_relocation false}} ]
  
   (py/call-attr-kw utils "multi_gpu_model" [] {:model model :gpus gpus :cpu_merge cpu_merge :cpu_relocation cpu_relocation }))

(defn normalize 
  "Normalizes a Numpy array.

    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).

    # Returns
        A normalized copy of the array.
    "
  [ & {:keys [x axis order]
       :or {axis -1 order 2}} ]
  
   (py/call-attr-kw utils "normalize" [] {:x x :axis axis :order order }))

(defn plot-model 
  "Converts a Keras model to dot format and save to a file.

    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    "
  [ & {:keys [model to_file show_shapes show_layer_names rankdir]
       :or {to_file "model.png" show_shapes false show_layer_names true rankdir "TB"}} ]
  
   (py/call-attr-kw utils "plot_model" [] {:model model :to_file to_file :show_shapes show_shapes :show_layer_names show_layer_names :rankdir rankdir }))

(defn print-summary 
  "Prints a summary of a model.

    # Arguments
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
    "
  [ & {:keys [model line_length positions print_fn]} ]
   (py/call-attr-kw utils "print_summary" [] {:model model :line_length line_length :positions positions :print_fn print_fn }))

(defn serialize-keras-object 
  ""
  [ & {:keys [instance]} ]
   (py/call-attr-kw utils "serialize_keras_object" [] {:instance instance }))

(defn to-categorical 
  "Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    "
  [ & {:keys [y num_classes dtype]
       :or {dtype "float32"}} ]
  
   (py/call-attr-kw utils "to_categorical" [] {:y y :num_classes num_classes :dtype dtype }))
