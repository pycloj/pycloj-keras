(ns keras.backend
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce backend (import-module "keras.backend"))

(defn abs 
  "Element-wise absolute value.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "abs" [] {:x x }))

(defn all 
  "Bitwise reduction (logical AND).

    # Arguments
        x: Tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the logical and. If `None` (default), computes
            the logical and over all dimensions.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "all" [] {:x x :axis axis :keepdims keepdims }))

(defn any 
  "Bitwise reduction (logical OR).

    # Arguments
        x: Tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the logical or. If `None` (default), computes
            the logical or over all dimensions.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "any" [] {:x x :axis axis :keepdims keepdims }))

(defn arange 
  "Creates a 1D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the \"stop\" argument and \"start\" is 0.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.

    # Arguments
        start: Start value.
        stop: Stop value.
        step: Difference between two successive values.
        dtype: Integer dtype to use.

    # Returns
        An integer tensor.

    "
  [ & {:keys [start stop step dtype]
       :or {step 1 dtype "int32"}} ]
  
   (py/call-attr-kw backend "arange" [] {:start start :stop stop :step step :dtype dtype }))

(defn argmax 
  "Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    "
  [ & {:keys [x axis]
       :or {axis -1}} ]
  
   (py/call-attr-kw backend "argmax" [] {:x x :axis axis }))

(defn argmin 
  "Returns the index of the minimum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    "
  [ & {:keys [x axis]
       :or {axis -1}} ]
  
   (py/call-attr-kw backend "argmin" [] {:x x :axis axis }))

(defn backend 
  "Publicly accessible method
    for determining the current backend.

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    "
  [  ]
  (py/call-attr backend "backend"   ))

(defn batch-dot 
  "Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17], [53]]` which is the main diagonal
        of `x.dot(y.T)`, although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
        If `axes` is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in `x`'s shape and `y`'s shape:

        * `x.shape[0]` : 100 : append to output shape
        * `x.shape[1]` : 20 : do not append to output shape,
            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
        * `y.shape[0]` : 100 : do not append to output shape,
            always ignore first dimension of `y`
        * `y.shape[1]` : 30 : append to output shape
        * `y.shape[2]` : 20 : do not append to output shape,
            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
        `output_shape` = `(100, 30)`

    ```python
        >>> x_batch = K.ones(shape=(32, 20, 1))
        >>> y_batch = K.ones(shape=(32, 30, 20))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```
    "
  [ & {:keys [x y axes]} ]
   (py/call-attr-kw backend "batch_dot" [] {:x x :y y :axes axes }))

(defn batch-flatten 
  "Turn a nD tensor into a 2D tensor with same 0th dimension.

    In other words, it flattens each data samples of a batch.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "batch_flatten" [] {:x x }))

(defn batch-get-value 
  "Returns the value of more than one tensor variable.

    # Arguments
        ops: list of ops to run.

    # Returns
        A list of Numpy arrays.
    "
  [ & {:keys [ops]} ]
   (py/call-attr-kw backend "batch_get_value" [] {:ops ops }))

(defn batch-normalization 
  "Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        axis: Integer, the axis that should be normalized.
            (typically the features axis).
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    "
  [ & {:keys [x mean var beta gamma axis epsilon]
       :or {axis -1 epsilon 0.001}} ]
  
   (py/call-attr-kw backend "batch_normalization" [] {:x x :mean mean :var var :beta beta :gamma gamma :axis axis :epsilon epsilon }))

(defn batch-set-value 
  "Sets the values of many tensor variables at once.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    "
  [ & {:keys [tuples]} ]
   (py/call-attr-kw backend "batch_set_value" [] {:tuples tuples }))

(defn bias-add 
  "Adds a bias vector to a tensor.

    # Arguments
        x: Tensor or variable.
        bias: Bias tensor to add.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.

    # Returns
        Output tensor.

    # Raises
        ValueError: In one of the two cases below:
                    1. invalid `data_format` argument.
                    2. invalid bias shape.
                       the bias should be either a vector or
                       a tensor with ndim(x) - 1 dimension
    "
  [ & {:keys [x bias data_format]} ]
   (py/call-attr-kw backend "bias_add" [] {:x x :bias bias :data_format data_format }))

(defn binary-crossentropy 
  "Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    "
  [ & {:keys [target output from_logits]
       :or {from_logits false}} ]
  
   (py/call-attr-kw backend "binary_crossentropy" [] {:target target :output output :from_logits from_logits }))

(defn cast 
  "Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder((2, 3), dtype='float32')
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
    ```
    "
  [ & {:keys [x dtype]} ]
   (py/call-attr-kw backend "cast" [] {:x x :dtype dtype }))

(defn cast-to-floatx 
  "Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "cast_to_floatx" [] {:x x }))

(defn categorical-crossentropy 
  "Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    "
  [ & {:keys [target output from_logits axis]
       :or {from_logits false axis -1}} ]
  
   (py/call-attr-kw backend "categorical_crossentropy" [] {:target target :output output :from_logits from_logits :axis axis }))

(defn clear-session 
  "Destroys the current TF graph and creates a new one.

    Useful to avoid clutter from old models / layers.
    "
  [  ]
  (py/call-attr backend "clear_session"   ))

(defn clip 
  "Element-wise value clipping.

    # Arguments
        x: Tensor or variable.
        min_value: Python float or integer.
        max_value: Python float or integer.

    # Returns
        A tensor.
    "
  [ & {:keys [x min_value max_value]} ]
   (py/call-attr-kw backend "clip" [] {:x x :min_value min_value :max_value max_value }))

(defn concatenate 
  "Concatenates a list of tensors alongside the specified axis.

    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.

    # Returns
        A tensor.
    "
  [ & {:keys [tensors axis]
       :or {axis -1}} ]
  
   (py/call-attr-kw backend "concatenate" [] {:tensors tensors :axis axis }))

(defn constant 
  "Creates a constant tensor.

    # Arguments
        value: A constant value (or list)
        dtype: The type of the elements of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.

    # Returns
        A Constant Tensor.
    "
  [ & {:keys [value dtype shape name]} ]
   (py/call-attr-kw backend "constant" [] {:value value :dtype dtype :shape shape :name name }))

(defn conv1d 
  "1D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `\"same\"`, `\"causal\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        dilation_rate: integer dilate rate.

    # Returns
        A tensor, result of 1D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x kernel strides padding data_format dilation_rate]
       :or {strides 1 padding "valid" dilation_rate 1}} ]
  
   (py/call-attr-kw backend "conv1d" [] {:x x :kernel kernel :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv2d 
  "2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x kernel strides padding data_format dilation_rate]
       :or {strides (1, 1) padding "valid" dilation_rate (1, 1)}} ]
  
   (py/call-attr-kw backend "conv2d" [] {:x x :kernel kernel :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv2d-transpose 
  "2D deconvolution (i.e. transposed convolution).

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of transposed 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x kernel output_shape strides padding data_format dilation_rate]
       :or {strides (1, 1) padding "valid" dilation_rate (1, 1)}} ]
  
   (py/call-attr-kw backend "conv2d_transpose" [] {:x x :kernel kernel :output_shape output_shape :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv3d 
  "3D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 3 integers.

    # Returns
        A tensor, result of 3D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x kernel strides padding data_format dilation_rate]
       :or {strides (1, 1, 1) padding "valid" dilation_rate (1, 1, 1)}} ]
  
   (py/call-attr-kw backend "conv3d" [] {:x x :kernel kernel :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv3d-transpose 
  "3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, \"same\" or \"valid\".
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.

    # Returns
        A tensor, result of transposed 3D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x kernel output_shape strides padding data_format]
       :or {strides (1, 1, 1) padding "valid"}} ]
  
   (py/call-attr-kw backend "conv3d_transpose" [] {:x x :kernel kernel :output_shape output_shape :strides strides :padding padding :data_format data_format }))

(defn cos 
  "Computes cos of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "cos" [] {:x x }))

(defn count-params 
  "Returns the static number of elements in a Keras variable or tensor.

    # Arguments
        x: Keras variable or tensor.

    # Returns
        Integer, the number of elements in `x`, i.e., the product of the
        array's static dimensions.

    # Example
    ```python
        >>> kvar = K.zeros((2,3))
        >>> K.count_params(kvar)
        6
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "count_params" [] {:x x }))

(defn ctc-batch-cost 
  "Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    "
  [ & {:keys [y_true y_pred input_length label_length]} ]
   (py/call-attr-kw backend "ctc_batch_cost" [] {:y_true y_true :y_pred y_pred :input_length input_length :label_length label_length }))

(defn ctc-decode 
  "Decodes the output of a softmax.

    Can use either greedy search (also known as best path)
    or a constrained dictionary search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.

    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    "
  [ & {:keys [y_pred input_length greedy beam_width top_paths]
       :or {greedy true beam_width 100 top_paths 1}} ]
  
   (py/call-attr-kw backend "ctc_decode" [] {:y_pred y_pred :input_length input_length :greedy greedy :beam_width beam_width :top_paths top_paths }))

(defn ctc-label-dense-to-sparse 
  "Converts CTC labels from dense to sparse.

    # Arguments
        labels: dense CTC labels.
        label_lengths: length of the labels.

    # Returns
        A sparse tensor representation of the labels.
    "
  [ & {:keys [labels label_lengths]} ]
   (py/call-attr-kw backend "ctc_label_dense_to_sparse" [] {:labels labels :label_lengths label_lengths }))

(defn cumprod 
  "Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    "
  [ & {:keys [x axis]
       :or {axis 0}} ]
  
   (py/call-attr-kw backend "cumprod" [] {:x x :axis axis }))

(defn cumsum 
  "Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    "
  [ & {:keys [x axis]
       :or {axis 0}} ]
  
   (py/call-attr-kw backend "cumsum" [] {:x x :axis axis }))

(defn depthwise-conv2d 
  "2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        strides: strides tuple (length 2).
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x depthwise_kernel strides padding data_format dilation_rate]
       :or {strides (1, 1) padding "valid" dilation_rate (1, 1)}} ]
  
   (py/call-attr-kw backend "depthwise_conv2d" [] {:x x :depthwise_kernel depthwise_kernel :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn dot 
  "Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.

    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
    ```

    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "dot" [] {:x x :y y }))

(defn dropout 
  "Sets entries in `x` to zero at random, while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.

    # Returns
        A tensor.
    "
  [ & {:keys [x level noise_shape seed]} ]
   (py/call-attr-kw backend "dropout" [] {:x x :level level :noise_shape noise_shape :seed seed }))

(defn dtype 
  "Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32_ref'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32_ref'
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "dtype" [] {:x x }))

(defn elu 
  "Exponential linear unit.

    # Arguments
        x: A tensor or variable to compute the activation function for.
        alpha: A scalar, slope of negative section.

    # Returns
        A tensor.
    "
  [ & {:keys [x alpha]
       :or {alpha 1.0}} ]
  
   (py/call-attr-kw backend "elu" [] {:x x :alpha alpha }))

(defn epsilon 
  "Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    "
  [  ]
  (py/call-attr backend "epsilon"   ))

(defn equal 
  "Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "equal" [] {:x x :y y }))

(defn eval 
  "Evaluates the value of a variable.

    # Arguments
        x: A variable.

    # Returns
        A Numpy array.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "eval" [] {:x x }))

(defn exp 
  "Element-wise exponential.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "exp" [] {:x x }))

(defn expand-dims 
  "Adds a 1-sized dimension at index \"axis\".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    "
  [ & {:keys [x axis]
       :or {axis -1}} ]
  
   (py/call-attr-kw backend "expand_dims" [] {:x x :axis axis }))

(defn eye 
  "Instantiate an identity matrix and returns it.

    # Arguments
        size: Integer, number of rows/columns.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, an identity matrix.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.eye(3)
        >>> K.eval(kvar)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
    ```

    "
  [ & {:keys [size dtype name]} ]
   (py/call-attr-kw backend "eye" [] {:size size :dtype dtype :name name }))

(defn flatten 
  "Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "flatten" [] {:x x }))

(defn floatx 
  "Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    "
  [  ]
  (py/call-attr backend "floatx"   ))

(defn foldl 
  "Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[0]` in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Tensor with same type and shape as `initializer`.
    "
  [ & {:keys [fn elems initializer name]} ]
   (py/call-attr-kw backend "foldl" [] {:fn fn :elems elems :initializer initializer :name name }))

(defn foldr 
  "Reduce elems using fn to combine them from right to left.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[-1]` in case of None)
        name: A string name for the foldr node in the graph

    # Returns
        Tensor with same type and shape as `initializer`.
    "
  [ & {:keys [fn elems initializer name]} ]
   (py/call-attr-kw backend "foldr" [] {:fn fn :elems elems :initializer initializer :name name }))

(defn function 
  "Instantiates a Keras function.

    # Arguments
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: Passed to `tf.Session.run`.

    # Returns
        Output values as Numpy arrays.

    # Raises
        ValueError: if invalid kwargs are passed in.
    "
  [ & {:keys [inputs outputs updates]} ]
   (py/call-attr-kw backend "function" [] {:inputs inputs :outputs outputs :updates updates }))

(defn gather 
  "Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    "
  [ & {:keys [reference indices]} ]
   (py/call-attr-kw backend "gather" [] {:reference reference :indices indices }))

(defn get-session 
  "Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global Keras session.

    If no global Keras session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `K.set_session(sess)`.

    # Returns
        A TensorFlow session.
    "
  [  ]
  (py/call-attr backend "get_session"   ))

(defn get-uid 
  "Get the uid for the default graph.

    # Arguments
        prefix: An optional prefix of the graph.

    # Returns
        A unique identifier for the graph.
    "
  [ & {:keys [prefix]
       :or {prefix ""}} ]
  
   (py/call-attr-kw backend "get_uid" [] {:prefix prefix }))

(defn get-value 
  "Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "get_value" [] {:x x }))

(defn get-variable-shape 
  "Returns the shape of a variable.

    # Arguments
        x: A variable.

    # Returns
        A tuple of integers.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "get_variable_shape" [] {:x x }))

(defn gradients 
  "Returns the gradients of `loss` w.r.t. `variables`.

    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.

    # Returns
        A gradients tensor.
    "
  [ & {:keys [loss variables]} ]
   (py/call-attr-kw backend "gradients" [] {:loss loss :variables variables }))

(defn greater 
  "Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "greater" [] {:x x :y y }))

(defn greater-equal 
  "Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "greater_equal" [] {:x x :y y }))

(defn hard-sigmoid 
  "Segment-wise linear approximation of sigmoid.

    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "hard_sigmoid" [] {:x x }))

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
  
   (py/call-attr-kw backend "has_arg" [] {:fn fn :name name :accept_all accept_all }))

(defn identity 
  "Returns a tensor with the same content as the input tensor.

    # Arguments
        x: The input tensor.
        name: String, name for the variable to create.

    # Returns
        A tensor of the same shape, type and content.
    "
  [ & {:keys [x name]} ]
   (py/call-attr-kw backend "identity" [] {:x x :name name }))

(defn image-data-format 
  "Returns the default image data format convention ('channels_first' or 'channels_last').

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    "
  [  ]
  (py/call-attr backend "image_data_format"   ))

(defn image-dim-ordering 
  "Legacy getter for `image_data_format`.

    # Returns
        string, one of `'th'`, `'tf'`
    "
  [  ]
  (py/call-attr backend "image_dim_ordering"   ))

(defn in-test-phase 
  "Selects `x` in test phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in test phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

    # Returns
        Either `x` or `alt` based on `K.learning_phase`.
    "
  [ & {:keys [x alt training]} ]
   (py/call-attr-kw backend "in_test_phase" [] {:x x :alt alt :training training }))

(defn in-top-k 
  "Returns whether the `targets` are in the top `k` `predictions`.

    # Arguments
        predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
        targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.

    # Returns
        A 1D tensor of length `batch_size` and type `bool`.
        `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
        values of `predictions[i]`.
    "
  [ & {:keys [predictions targets k]} ]
   (py/call-attr-kw backend "in_top_k" [] {:predictions predictions :targets targets :k k }))

(defn in-train-phase 
  "Selects `x` in train phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in train phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

    # Returns
        Either `x` or `alt` based on the `training` flag.
        the `training` flag defaults to `K.learning_phase()`.
    "
  [ & {:keys [x alt training]} ]
   (py/call-attr-kw backend "in_train_phase" [] {:x x :alt alt :training training }))

(defn int-shape 
  "Returns the shape of tensor or variable as a tuple of int or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "int_shape" [] {:x x }))

(defn is-keras-tensor 
  "Returns whether `x` is a Keras tensor.

    A \"Keras tensor\" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.

    # Arguments
        x: A candidate tensor.

    # Returns
        A boolean: Whether the argument is a Keras tensor.

    # Raises
        ValueError: In case `x` is not a symbolic tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> from keras.layers import Input, Dense
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
        ValueError
        >>> k_var = tf.placeholder('float32', shape=(1,1))
        >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
        True
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "is_keras_tensor" [] {:x x }))

(defn is-placeholder 
  "Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "is_placeholder" [] {:x x }))

(defn is-sparse 
  "Returns whether a tensor is a sparse tensor.

    # Arguments
        tensor: A tensor instance.

    # Returns
        A boolean.

    # Example
    ```python
        >>> from keras import backend as K
        >>> a = K.placeholder((2, 2), sparse=False)
        >>> print(K.is_sparse(a))
        False
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
    ```
    "
  [ & {:keys [tensor]} ]
   (py/call-attr-kw backend "is_sparse" [] {:tensor tensor }))

(defn is-tensor 
  ""
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "is_tensor" [] {:x x }))

(defn l2-normalize 
  "Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.
    "
  [ & {:keys [x axis]} ]
   (py/call-attr-kw backend "l2_normalize" [] {:x x :axis axis }))

(defn learning-phase 
  "Returns the learning phase flag.

    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.

    # Returns
        Learning phase (scalar integer tensor or Python integer).
    "
  [  ]
  (py/call-attr backend "learning_phase"   ))

(defn less 
  "Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "less" [] {:x x :y y }))

(defn less-equal 
  "Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "less_equal" [] {:x x :y y }))

(defn local-conv1d 
  "Apply 1D conv with un-shared weights.

    # Arguments
        inputs: 3D tensor with shape: (batch_size, steps, input_dim)
        kernel: the unshared weight for convolution,
                with shape (output_length, feature_dim, filters)
        kernel_size: a tuple of a single integer,
                     specifying the length of the 1D convolution window
        strides: a tuple of a single integer,
                 specifying the stride length of the convolution
        data_format: the data format, channels_first or channels_last

    # Returns
        the tensor after 1d conv with un-shared weights, with shape (batch_size, output_length, filters)

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [inputs kernel kernel_size strides data_format]} ]
   (py/call-attr-kw backend "local_conv1d" [] {:inputs inputs :kernel kernel :kernel_size kernel_size :strides strides :data_format data_format }))

(defn local-conv2d 
  "Apply 2D conv with un-shared weights.

    # Arguments
        inputs: 4D tensor with shape:
                (batch_size, filters, new_rows, new_cols)
                if data_format='channels_first'
                or 4D tensor with shape:
                (batch_size, new_rows, new_cols, filters)
                if data_format='channels_last'.
        kernel: the unshared weight for convolution,
                with shape (output_items, feature_dim, filters)
        kernel_size: a tuple of 2 integers, specifying the
                     width and height of the 2D convolution window.
        strides: a tuple of 2 integers, specifying the strides
                 of the convolution along the width and height.
        output_shape: a tuple with (output_row, output_col)
        data_format: the data format, channels_first or channels_last

    # Returns
        A 4d tensor with shape:
        (batch_size, filters, new_rows, new_cols)
        if data_format='channels_first'
        or 4D tensor with shape:
        (batch_size, new_rows, new_cols, filters)
        if data_format='channels_last'.

    # Raises
        ValueError: if `data_format` is neither
                    `channels_last` or `channels_first`.
    "
  [ & {:keys [inputs kernel kernel_size strides output_shape data_format]} ]
   (py/call-attr-kw backend "local_conv2d" [] {:inputs inputs :kernel kernel :kernel_size kernel_size :strides strides :output_shape output_shape :data_format data_format }))

(defn log 
  "Element-wise log.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "log" [] {:x x }))

(defn logsumexp 
  "Computes log(sum(exp(elements across dimensions of a tensor))).

    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.

    # Arguments
        x: A tensor or variable.
        axis: axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the logsumexp. If `None` (default), computes
            the logsumexp over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.

    # Returns
        The reduced tensor.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "logsumexp" [] {:x x :axis axis :keepdims keepdims }))

(defn manual-variable-initialization 
  "Sets the manual variable initialization flag.

    This boolean flag determines whether
    variables should be initialized
    as they are instantiated (default), or if
    the user should handle the initialization
    (e.g. via `tf.initialize_all_variables()`).

    # Arguments
        value: Python boolean.
    "
  [ & {:keys [value]} ]
   (py/call-attr-kw backend "manual_variable_initialization" [] {:value value }))

(defn map-fn 
  "Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph
        dtype: Output data type.

    # Returns
        Tensor with dtype `dtype`.
    "
  [ & {:keys [fn elems name dtype]} ]
   (py/call-attr-kw backend "map_fn" [] {:fn fn :elems elems :name name :dtype dtype }))

(defn max 
  "Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to find maximum values. If `None` (default), finds the
            maximum over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "max" [] {:x x :axis axis :keepdims keepdims }))

(defn maximum 
  "Element-wise maximum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "maximum" [] {:x x :y y }))

(defn mean 
  "Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the mean. If `None` (default), computes
            the mean over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "mean" [] {:x x :axis axis :keepdims keepdims }))

(defn min 
  "Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to find minimum values. If `None` (default), finds the
            minimum over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "min" [] {:x x :axis axis :keepdims keepdims }))

(defn minimum 
  "Element-wise minimum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "minimum" [] {:x x :y y }))

(defn moving-average-update 
  "Compute the moving average of a variable.

    # Arguments
        x: A `Variable`.
        value: A tensor with the same shape as `x`.
        momentum: The moving average momentum.

    # Returns
        An operation to update the variable.
    "
  [ & {:keys [x value momentum]} ]
   (py/call-attr-kw backend "moving_average_update" [] {:x x :value value :momentum momentum }))

(defn ndim 
  "Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "ndim" [] {:x x }))

(defn normalize-batch-in-training 
  "Computes mean and std for batch then apply batch_normalization on batch.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    "
  [ & {:keys [x gamma beta reduction_axes epsilon]
       :or {epsilon 0.001}} ]
  
   (py/call-attr-kw backend "normalize_batch_in_training" [] {:x x :gamma gamma :beta beta :reduction_axes reduction_axes :epsilon epsilon }))

(defn normalize-data-format 
  "Checks that the value correspond to a valid data format.

    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    "
  [ & {:keys [value]} ]
   (py/call-attr-kw backend "normalize_data_format" [] {:value value }))

(defn not-equal 
  "Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    "
  [ & {:keys [x y]} ]
   (py/call-attr-kw backend "not_equal" [] {:x x :y y }))

(defn one-hot 
  "Computes the one-hot representation of an integer tensor.

    # Arguments
        indices: nD integer tensor of shape
            `(batch_size, dim1, dim2, ... dim(n-1))`
        num_classes: Integer, number of classes to consider.

    # Returns
        (n + 1)D one hot representation of the input
        with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
    "
  [ & {:keys [indices num_classes]} ]
   (py/call-attr-kw backend "one_hot" [] {:indices indices :num_classes num_classes }))

(defn ones 
  "Instantiates an all-ones variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, filled with `1.0`.
        Note that if `shape` was symbolic, we cannot return a variable,
        and will return a dynamically-shaped tensor instead.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.ones((3,4))
        >>> K.eval(kvar)
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)
    ```
    "
  [ & {:keys [shape dtype name]} ]
   (py/call-attr-kw backend "ones" [] {:shape shape :dtype dtype :name name }))

(defn ones-like 
  "Instantiates an all-ones variable of the same shape as another tensor.

    # Arguments
        x: Keras variable or tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.

    # Returns
        A Keras variable with the shape of x filled with ones.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_ones = K.ones_like(kvar)
        >>> K.eval(kvar_ones)
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
    ```
    "
  [ & {:keys [x dtype name]} ]
   (py/call-attr-kw backend "ones_like" [] {:x x :dtype dtype :name name }))

(defn permute-dimensions 
  "Permutes axes in a tensor.

    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.

    # Returns
        A tensor.
    "
  [ & {:keys [x pattern]} ]
   (py/call-attr-kw backend "permute_dimensions" [] {:x x :pattern pattern }))

(defn placeholder 
  "Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        sparse: Boolean, whether the placeholder should have a sparse type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph._keras_shape
        (2, 4, 5)
        >>> input_ph
        <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
    ```
    "
  [ & {:keys [shape ndim dtype sparse name]
       :or {sparse false}} ]
  
   (py/call-attr-kw backend "placeholder" [] {:shape shape :ndim ndim :dtype dtype :sparse sparse :name name }))

(defn pool2d 
  "2D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        pool_mode: string, `\"max\"` or `\"avg\"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is neither `\"channels_last\"` or `\"channels_first\"`.
        ValueError: if `pool_mode` is neither `\"max\"` or `\"avg\"`.
    "
  [ & {:keys [x pool_size strides padding data_format pool_mode]
       :or {strides (1, 1) padding "valid" pool_mode "max"}} ]
  
   (py/call-attr-kw backend "pool2d" [] {:x x :pool_size pool_size :strides strides :padding padding :data_format data_format :pool_mode pool_mode }))

(defn pool3d 
  "3D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        pool_mode: string, `\"max\"` or `\"avg\"`.

    # Returns
        A tensor, result of 3D pooling.

    # Raises
        ValueError: if `data_format` is neither `\"channels_last\"` or `\"channels_first\"`.
        ValueError: if `pool_mode` is neither `\"max\"` or `\"avg\"`.
    "
  [ & {:keys [x pool_size strides padding data_format pool_mode]
       :or {strides (1, 1, 1) padding "valid" pool_mode "max"}} ]
  
   (py/call-attr-kw backend "pool3d" [] {:x x :pool_size pool_size :strides strides :padding padding :data_format data_format :pool_mode pool_mode }))

(defn pow 
  "Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    "
  [ & {:keys [x a]} ]
   (py/call-attr-kw backend "pow" [] {:x x :a a }))

(defn print-tensor 
  "Prints `message` and the tensor value when evaluated.

     Note that `print_tensor` returns a new tensor identical to `x`
     which should be used in the following code. Otherwise the
     print operation is not taken into account during evaluation.

     # Example
     ```python
         >>> x = K.print_tensor(x, message=\"x is: \")
     ```

    # Arguments
        x: Tensor to print.
        message: Message to print jointly with the tensor.

    # Returns
        The same tensor `x`, unchanged.
    "
  [ & {:keys [x message]
       :or {message ""}} ]
  
   (py/call-attr-kw backend "print_tensor" [] {:x x :message message }))

(defn prod 
  "Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the product. If `None` (default), computes
            the product over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "prod" [] {:x x :axis axis :keepdims keepdims }))

(defn random-binomial 
  "Returns a tensor with random binomial distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    "
  [ & {:keys [shape p dtype seed]
       :or {p 0.0}} ]
  
   (py/call-attr-kw backend "random_binomial" [] {:shape shape :p p :dtype dtype :seed seed }))

(defn random-normal 
  "Returns a tensor with normal distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: A float, mean of the normal distribution to draw samples.
        stddev: A float, standard deviation of the normal distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    "
  [ & {:keys [shape mean stddev dtype seed]
       :or {mean 0.0 stddev 1.0}} ]
  
   (py/call-attr-kw backend "random_normal" [] {:shape shape :mean mean :stddev stddev :dtype dtype :seed seed }))

(defn random-normal-variable 
  "Instantiates a variable with values drawn from a normal distribution.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        mean: Float, mean of the normal distribution.
        scale: Float, standard deviation of the normal distribution.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        # TensorFlow example
        >>> kvar = K.random_normal_variable((2,3), 0, 1)
        >>> kvar
        <tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
        >>> K.eval(kvar)
        array([[ 1.19591331,  0.68685907, -0.63814116],
               [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
    ```
    "
  [ & {:keys [shape mean scale dtype name seed]} ]
   (py/call-attr-kw backend "random_normal_variable" [] {:shape shape :mean mean :scale scale :dtype dtype :name name :seed seed }))

(defn random-uniform 
  "Returns a tensor with uniform distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    "
  [ & {:keys [shape minval maxval dtype seed]
       :or {minval 0.0 maxval 1.0}} ]
  
   (py/call-attr-kw backend "random_uniform" [] {:shape shape :minval minval :maxval maxval :dtype dtype :seed seed }))

(defn random-uniform-variable 
  "Instantiates a variable with values drawn from a uniform distribution.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output interval.
        high: Float, upper boundary of the output interval.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        # TensorFlow example
        >>> kvar = K.random_uniform_variable((2,3), 0, 1)
        >>> kvar
        <tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
        >>> K.eval(kvar)
        array([[ 0.10940075,  0.10047495,  0.476143  ],
               [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
    ```
    "
  [ & {:keys [shape low high dtype name seed]} ]
   (py/call-attr-kw backend "random_uniform_variable" [] {:shape shape :low low :high high :dtype dtype :name name :seed seed }))

(defn relu 
  "Rectified linear unit.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.

    # Returns
        A tensor.
    "
  [ & {:keys [x alpha max_value threshold]
       :or {alpha 0.0 threshold 0.0}} ]
  
   (py/call-attr-kw backend "relu" [] {:x x :alpha alpha :max_value max_value :threshold threshold }))

(defn repeat 
  "Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    "
  [ & {:keys [x n]} ]
   (py/call-attr-kw backend "repeat" [] {:x x :n n }))

(defn repeat-elements 
  "Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.

    # Returns
        A tensor.
    "
  [ & {:keys [x rep axis]} ]
   (py/call-attr-kw backend "repeat_elements" [] {:x x :rep rep :axis axis }))

(defn reset-uids 
  "Resets graph identifiers.
    "
  [  ]
  (py/call-attr backend "reset_uids"   ))

(defn reshape 
  "Reshapes a tensor to the specified shape.

    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.

    # Returns
        A tensor.
    "
  [ & {:keys [x shape]} ]
   (py/call-attr-kw backend "reshape" [] {:x x :shape shape }))

(defn resize-images 
  "Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        interpolation: A string, one of `nearest` or `bilinear`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `\"channels_last\"` or `\"channels_first\"`.
    "
  [ & {:keys [x height_factor width_factor data_format interpolation]
       :or {interpolation "nearest"}} ]
  
   (py/call-attr-kw backend "resize_images" [] {:x x :height_factor height_factor :width_factor width_factor :data_format data_format :interpolation interpolation }))

(defn resize-volumes 
  "Resizes the volume contained in a 5D tensor.

    # Arguments
        x: Tensor or variable to resize.
        depth_factor: Positive integer.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `\"channels_last\"` or `\"channels_first\"`.
    "
  [ & {:keys [x depth_factor height_factor width_factor data_format]} ]
   (py/call-attr-kw backend "resize_volumes" [] {:x x :depth_factor depth_factor :height_factor height_factor :width_factor width_factor :data_format data_format }))

(defn reverse 
  "Reverses a tensor along the specified axes.

    # Arguments
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.

    # Returns
        A tensor.
    "
  [ & {:keys [x axes]} ]
   (py/call-attr-kw backend "reverse" [] {:x x :axes axes }))

(defn rnn 
  "Iterates over the time dimension of a tensor.

    # Arguments
        step_function:
            Parameters:
                inputs: Tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: List of tensors.
            Returns:
                outputs: Tensor with shape (samples, ...) (no time dimension),
                new_states: List of tensors, same length and shapes
                    as 'states'.
        inputs: Tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        initial_states: Tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: Boolean. If True, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
        mask: Binary tensor with shape (samples, time),
            with a zero for every element that is masked.
        constants: A list of constant values passed at each step.
        unroll: Whether to unroll the RNN or to use a symbolic loop
            (`while_loop` or `scan` depending on backend).
        input_length: Static number of timesteps in the input.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

        last_output: The latest output of the rnn, of shape `(samples, ...)`
        outputs: Tensor with shape `(samples, time, ...)` where each
            entry `outputs[s, t]` is the output of the step function
            at time `t` for sample `s`.
        new_states: List of tensors, latest states returned by
            the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: If input dimension is less than 3.
        ValueError: If `unroll` is `True`
            but input timestep is not a fixed number.
        ValueError: If `mask` is provided (not `None`)
            but states is not provided (`len(states)` == 0).
    "
  [ & {:keys [step_function inputs initial_states go_backwards mask constants unroll input_length]
       :or {go_backwards false unroll false}} ]
  
   (py/call-attr-kw backend "rnn" [] {:step_function step_function :inputs inputs :initial_states initial_states :go_backwards go_backwards :mask mask :constants constants :unroll unroll :input_length input_length }))

(defn round 
  "Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is \"half to even\".

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "round" [] {:x x }))

(defn separable-conv1d 
  "1D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: stride integer.
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        dilation_rate: integer dilation rate.

    # Returns
        Output tensor.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x depthwise_kernel pointwise_kernel strides padding data_format dilation_rate]
       :or {strides 1 padding "valid" dilation_rate 1}} ]
  
   (py/call-attr-kw backend "separable_conv1d" [] {:x x :depthwise_kernel depthwise_kernel :pointwise_kernel pointwise_kernel :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn separable-conv2d 
  "2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: strides tuple (length 2).
        padding: string, `\"same\"` or `\"valid\"`.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: If `data_format` is neither
            `\"channels_last\"` nor `\"channels_first\"`.
    "
  [ & {:keys [x depthwise_kernel pointwise_kernel strides padding data_format dilation_rate]
       :or {strides (1, 1) padding "valid" dilation_rate (1, 1)}} ]
  
   (py/call-attr-kw backend "separable_conv2d" [] {:x x :depthwise_kernel depthwise_kernel :pointwise_kernel pointwise_kernel :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn set-epsilon 
  "Sets the value of the fuzz factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-07
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    "
  [ & {:keys [e]} ]
   (py/call-attr-kw backend "set_epsilon" [] {:e e }))

(defn set-floatx 
  "Sets the default float type.

    # Arguments
        floatx: String, 'float16', 'float32', or 'float64'.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    "
  [ & {:keys [floatx]} ]
   (py/call-attr-kw backend "set_floatx" [] {:floatx floatx }))

(defn set-image-data-format 
  "Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```
    "
  [ & {:keys [data_format]} ]
   (py/call-attr-kw backend "set_image_data_format" [] {:data_format data_format }))

(defn set-image-dim-ordering 
  "Legacy setter for `image_data_format`.

    # Arguments
        dim_ordering: string. `tf` or `th`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```

    # Raises
        ValueError: if `dim_ordering` is invalid.
    "
  [ & {:keys [dim_ordering]} ]
   (py/call-attr-kw backend "set_image_dim_ordering" [] {:dim_ordering dim_ordering }))

(defn set-learning-phase 
  "Sets the learning phase to a fixed value.

    # Arguments
        value: Learning phase value, either 0 or 1 (integers).

    # Raises
        ValueError: if `value` is neither `0` nor `1`.
    "
  [ & {:keys [value]} ]
   (py/call-attr-kw backend "set_learning_phase" [] {:value value }))

(defn set-session 
  "Sets the global TensorFlow session.

    # Arguments
        session: A TF Session.
    "
  [ & {:keys [session]} ]
   (py/call-attr-kw backend "set_session" [] {:session session }))

(defn set-value 
  "Sets the value of a variable, from a Numpy array.

    # Arguments
        x: Tensor to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    "
  [ & {:keys [x value]} ]
   (py/call-attr-kw backend "set_value" [] {:x x :value value }))

(defn shape 
  "Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```python
        # TensorFlow example
        >>> from keras import backend as K
        >>> tf_session = K.get_session()
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
        >>> K.shape(inputs)
        <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
        # To get integer shape (Instead, you can use K.int_shape(x))
        >>> K.shape(kvar).eval(session=tf_session)
        array([2, 2], dtype=int32)
        >>> K.shape(inputs).eval(session=tf_session)
        array([2, 4, 5], dtype=int32)
    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "shape" [] {:x x }))

(defn sigmoid 
  "Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "sigmoid" [] {:x x }))

(defn sign 
  "Element-wise sign.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "sign" [] {:x x }))

(defn sin 
  "Computes sin of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "sin" [] {:x x }))

(defn slice 
  "Extracts a slice from a tensor.

    # Arguments
        x: Input tensor.
        start: Integer list/tuple or tensor
            indicating the start indices of the slice
            along each axis.
        size: Integer list/tuple or tensor
            indicating how many dimensions to slice
            along each axis.

    # Returns
        Tensor `x[start[0]: start[0] + size[0],
                  ...,
                  start[-1]: start[-1] + size[-1]]`
    "
  [ & {:keys [x start size]} ]
   (py/call-attr-kw backend "slice" [] {:x x :start start :size size }))

(defn softmax 
  "Softmax of a tensor.

    # Arguments
        x: A tensor or variable.
        axis: The dimension softmax would be performed on.
            The default is -1 which indicates the last dimension.

    # Returns
        A tensor.
    "
  [ & {:keys [x axis]
       :or {axis -1}} ]
  
   (py/call-attr-kw backend "softmax" [] {:x x :axis axis }))

(defn softplus 
  "Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "softplus" [] {:x x }))

(defn softsign 
  "Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "softsign" [] {:x x }))

(defn sparse-categorical-crossentropy 
  "Categorical crossentropy with integer targets.

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    "
  [ & {:keys [target output from_logits axis]
       :or {from_logits false axis -1}} ]
  
   (py/call-attr-kw backend "sparse_categorical_crossentropy" [] {:target target :output output :from_logits from_logits :axis axis }))

(defn spatial-2d-padding 
  "Pads the 2nd and 3rd dimensions of a 4D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `data_format` is neither `\"channels_last\"` or `\"channels_first\"`.
    "
  [ & {:keys [x padding data_format]
       :or {padding ((1, 1), (1, 1))}} ]
  
   (py/call-attr-kw backend "spatial_2d_padding" [] {:x x :padding padding :data_format data_format }))

(defn spatial-3d-padding 
  "Pads 5D tensor with zeros along the depth, height, width dimensions.

    Pads these dimensions with respectively
    \"padding[0]\", \"padding[1]\" and \"padding[2]\" zeros left and right.

    For 'channels_last' data_format,
    the 2nd, 3rd and 4th dimension will be padded.
    For 'channels_first' data_format,
    the 3rd, 4th and 5th dimension will be padded.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 3 tuples, padding pattern.
        data_format: string, `\"channels_last\"` or `\"channels_first\"`.

    # Returns
        A padded 5D tensor.

    # Raises
        ValueError: if `data_format` is neither `\"channels_last\"` or `\"channels_first\"`.

    "
  [ & {:keys [x padding data_format]
       :or {padding ((1, 1), (1, 1), (1, 1))}} ]
  
   (py/call-attr-kw backend "spatial_3d_padding" [] {:x x :padding padding :data_format data_format }))

(defn sqrt 
  "Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "sqrt" [] {:x x }))

(defn square 
  "Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "square" [] {:x x }))

(defn squeeze 
  "Removes a 1-dimension from the tensor at index \"axis\".

    # Arguments
        x: A tensor or variable.
        axis: Axis to drop.

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    "
  [ & {:keys [x axis]} ]
   (py/call-attr-kw backend "squeeze" [] {:x x :axis axis }))

(defn stack 
  "Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: List of tensors.
        axis: Axis along which to perform stacking.

    # Returns
        A tensor.
    "
  [ & {:keys [x axis]
       :or {axis 0}} ]
  
   (py/call-attr-kw backend "stack" [] {:x x :axis axis }))

(defn std 
  "Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the standard deviation. If `None` (default),
            computes the standard deviation over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "std" [] {:x x :axis axis :keepdims keepdims }))

(defn stop-gradient 
  "Returns `variables` but with zero gradient w.r.t. every other variable.

    # Arguments
        variables: tensor or list of tensors to consider constant with respect
            to any other variable.

    # Returns
        A single tensor or a list of tensors (depending on the passed argument)
            that has constant gradient with respect to any other variable.
    "
  [ & {:keys [variables]} ]
   (py/call-attr-kw backend "stop_gradient" [] {:variables variables }))

(defn sum 
  "Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to sum over. If `None` (default), sums over all
            dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "sum" [] {:x x :axis axis :keepdims keepdims }))

(defn switch 
  "Switches between two operations depending on a scalar value.

    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: tensor (`int` or `bool`).
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.

    # Raises
        ValueError: If rank of `condition` is greater than rank of expressions.
    "
  [ & {:keys [condition then_expression else_expression]} ]
   (py/call-attr-kw backend "switch" [] {:condition condition :then_expression then_expression :else_expression else_expression }))

(defn tanh 
  "Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "tanh" [] {:x x }))

(defn temporal-padding 
  "Pads the middle dimension of a 3D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.

    # Returns
        A padded 3D tensor.
    "
  [ & {:keys [x padding]
       :or {padding (1, 1)}} ]
  
   (py/call-attr-kw backend "temporal_padding" [] {:x x :padding padding }))

(defn tile 
  "Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    "
  [ & {:keys [x n]} ]
   (py/call-attr-kw backend "tile" [] {:x x :n n }))

(defn to-dense 
  "Converts a sparse tensor into a dense tensor and returns it.

    # Arguments
        tensor: A tensor instance (potentially sparse).

    # Returns
        A dense tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
        >>> c = K.to_dense(b)
        >>> print(K.is_sparse(c))
        False
    ```
    "
  [ & {:keys [tensor]} ]
   (py/call-attr-kw backend "to_dense" [] {:tensor tensor }))

(defn transpose 
  "Transposes a tensor and returns it.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.

    # Examples
    ```python
        >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
        >>> K.eval(var)
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> var_transposed = K.transpose(var)
        >>> K.eval(var_transposed)
        array([[ 1.,  4.],
               [ 2.,  5.],
               [ 3.,  6.]], dtype=float32)
    ```

    ```python
        >>> inputs = K.placeholder((2, 3))
        >>> inputs
        <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
        >>> input_transposed = K.transpose(inputs)
        >>> input_transposed
        <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

    ```
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw backend "transpose" [] {:x x }))

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
  [ & {:keys [shape target_format spatial_axes]} ]
   (py/call-attr-kw backend "transpose_shape" [] {:shape shape :target_format target_format :spatial_axes spatial_axes }))

(defn truncated-normal 
  "Returns a tensor with truncated random normal distribution of values.

    The generated values follow a normal distribution
    with specified mean and standard deviation,
    except that values whose magnitude is more than
    two standard deviations from the mean are dropped and re-picked.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: Mean of the values.
        stddev: Standard deviation of the values.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    "
  [ & {:keys [shape mean stddev dtype seed]
       :or {mean 0.0 stddev 1.0}} ]
  
   (py/call-attr-kw backend "truncated_normal" [] {:shape shape :mean mean :stddev stddev :dtype dtype :seed seed }))

(defn update 
  "Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    "
  [ & {:keys [x new_x]} ]
   (py/call-attr-kw backend "update" [] {:x x :new_x new_x }))

(defn update-add 
  "Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    "
  [ & {:keys [x increment]} ]
   (py/call-attr-kw backend "update_add" [] {:x x :increment increment }))

(defn update-sub 
  "Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    "
  [ & {:keys [x decrement]} ]
   (py/call-attr-kw backend "update_sub" [] {:x x :decrement decrement }))

(defn var 
  "Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the variance. If `None` (default), computes
            the variance over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    "
  [ & {:keys [x axis keepdims]
       :or {keepdims false}} ]
  
   (py/call-attr-kw backend "var" [] {:x x :axis axis :keepdims keepdims }))

(defn variable 
  "Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.

    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    "
  [ & {:keys [value dtype name constraint]} ]
   (py/call-attr-kw backend "variable" [] {:value value :dtype dtype :name name :constraint constraint }))

(defn zeros 
  "Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.
        Note that if `shape` was symbolic, we cannot return a variable,
        and will return a dynamically-shaped tensor instead.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.zeros((3,4))
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
    ```
    "
  [ & {:keys [shape dtype name]} ]
   (py/call-attr-kw backend "zeros" [] {:shape shape :dtype dtype :name name }))

(defn zeros-like 
  "Instantiates an all-zeros variable of the same shape as another tensor.

    # Arguments
        x: Keras variable or Keras tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.

    # Returns
        A Keras variable with the shape of x filled with zeros.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_zeros = K.zeros_like(kvar)
        >>> K.eval(kvar_zeros)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    "
  [ & {:keys [x dtype name]} ]
   (py/call-attr-kw backend "zeros_like" [] {:x x :dtype dtype :name name }))
