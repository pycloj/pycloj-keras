(ns keras.utils.conv-utils
  "Utilities used in convolutional layers.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce conv-utils (import-module "keras.utils.conv_utils"))

(defn conv-input-length 
  "Determines input length of a convolution given output length.

    # Arguments
        output_length: integer.
        filter_size: integer.
        padding: one of `\"same\"`, `\"valid\"`, `\"full\"`.
        stride: integer.

    # Returns
        The input length (integer).
    "
  [ & {:keys [output_length filter_size padding stride]} ]
   (py/call-attr-kw conv-utils "conv_input_length" [] {:output_length output_length :filter_size filter_size :padding padding :stride stride }))

(defn conv-output-length 
  "Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `\"same\"`, `\"valid\"`, `\"full\"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    "
  [ & {:keys [input_length filter_size padding stride dilation]
       :or {dilation 1}} ]
  
   (py/call-attr-kw conv-utils "conv_output_length" [] {:input_length input_length :filter_size filter_size :padding padding :stride stride :dilation dilation }))

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
  [ & {:keys [kernel]} ]
   (py/call-attr-kw conv-utils "convert_kernel" [] {:kernel kernel }))

(defn deconv-length 
  "Determines output length of a transposed convolution given input length.

    # Arguments
        dim_size: Integer, the input length.
        stride_size: Integer, the stride along the dimension of `dim_size`.
        kernel_size: Integer, the kernel size along the dimension of
            `dim_size`.
        padding: One of `\"same\"`, `\"valid\"`, `\"full\"`.
        output_padding: Integer, amount of padding along the output dimension,
            Can be set to `None` in which case the output length is inferred.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    "
  [ & {:keys [dim_size stride_size kernel_size padding output_padding dilation]
       :or {dilation 1}} ]
  
   (py/call-attr-kw conv-utils "deconv_length" [] {:dim_size dim_size :stride_size stride_size :kernel_size kernel_size :padding padding :output_padding output_padding :dilation dilation }))

(defn normalize-padding 
  ""
  [ & {:keys [value]} ]
   (py/call-attr-kw conv-utils "normalize_padding" [] {:value value }))

(defn normalize-tuple 
  "Transforms a single int or iterable of ints into an int tuple.

    # Arguments
        value: The value to validate and convert. Could be an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. `strides` or
          `kernel_size`. This is only used to format error messages.

    # Returns
        A tuple of n integers.

    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    "
  [ & {:keys [value n name]} ]
   (py/call-attr-kw conv-utils "normalize_tuple" [] {:value value :n n :name name }))
