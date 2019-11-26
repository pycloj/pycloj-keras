(ns keras.utils.layer-utils
  "Utilities related to layer/model functionality.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce layer-utils (import-module "keras.utils.layer_utils"))

(defn convert-all-kernels-in-model 
  "Converts all convolution kernels in a model from Theano to TensorFlow.

    Also works from TensorFlow to Theano.

    # Arguments
        model: target model for the conversion.
    "
  [ model ]
  (py/call-attr layer-utils "convert_all_kernels_in_model"  model ))

(defn convert-dense-weights-data-format 
  "Utility useful when changing a convnet's `data_format`.

    When porting the weights of a convnet from one data format to the other,
    if the convnet includes a `Flatten` layer
    (applied to the last convolutional feature map)
    followed by a `Dense` layer, the weights of that `Dense` layer
    should be updated to reflect the new dimension ordering.

    # Arguments
        dense: The target `Dense` layer.
        previous_feature_map_shape: A shape tuple of 3 integers,
            e.g. `(512, 7, 7)`. The shape of the convolutional
            feature map right before the `Flatten` layer that
            came before the target `Dense` layer.
        target_data_format: One of \"channels_last\", \"channels_first\".
            Set it \"channels_last\"
            if converting a \"channels_first\" model to \"channels_last\",
            or reciprocally.
    "
  [dense previous_feature_map_shape & {:keys [target_data_format]
                       :or {target_data_format "channels_first"}} ]
    (py/call-attr-kw layer-utils "convert_dense_weights_data_format" [dense previous_feature_map_shape] {:target_data_format target_data_format }))

(defn convert-kernel 
  "Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    # Arguments
        kernel: Numpy array (3D, 4D or 5D).

    # Returns
        The converted kernel.

    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    "
  [ kernel ]
  (py/call-attr layer-utils "convert_kernel"  kernel ))

(defn count-params 
  "Count the total number of scalars composing the weights.

    # Arguments
        weights: An iterable containing the weights on which to compute params

    # Returns
        The total number of scalars composing the weights
    "
  [ weights ]
  (py/call-attr layer-utils "count_params"  weights ))
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
  [tensor  & {:keys [layer node_index]} ]
    (py/call-attr-kw layer-utils "get_source_inputs" [tensor] {:layer layer :node_index node_index }))
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
  [model  & {:keys [line_length positions print_fn]} ]
    (py/call-attr-kw layer-utils "print_summary" [model] {:line_length line_length :positions positions :print_fn print_fn }))
