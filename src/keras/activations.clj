(ns keras.activations
  "Built-in activation functions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce activations (import-module "keras.activations"))
(defn deserialize 
  ""
  [name  & {:keys [custom_objects]} ]
    (py/call-attr-kw activations "deserialize" [name] {:custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw activations "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn elu 
  "Exponential linear unit.

    # Arguments
        x: Input tensor.
        alpha: A scalar, slope of negative section.

    # Returns
        The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.

    # References
        - [Fast and Accurate Deep Network Learning by Exponential
           Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    "
  [x & {:keys [alpha]
                       :or {alpha 1.0}} ]
    (py/call-attr-kw activations "elu" [x] {:alpha alpha }))

(defn exponential 
  "Exponential (base e) activation function.

    # Arguments
        x: Input tensor.

    # Returns
        Exponential activation: `exp(x)`.
    "
  [ x ]
  (py/call-attr activations "exponential"  x ))

(defn get 
  "Get the `identifier` activation function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The activation function, `linear` if `identifier` is None.

    # Raises
        ValueError if unknown identifier
    "
  [ identifier ]
  (py/call-attr activations "get"  identifier ))

(defn hard-sigmoid 
  "Hard sigmoid activation function.

    Faster to compute than sigmoid activation.

    # Arguments
        x: Input tensor.

    # Returns
        Hard sigmoid activation:

        - `0` if `x < -2.5`
        - `1` if `x > 2.5`
        - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
    "
  [ x ]
  (py/call-attr activations "hard_sigmoid"  x ))

(defn linear 
  "Linear (i.e. identity) activation function.

    # Arguments
        x: Input tensor.

    # Returns
        Input tensor, unchanged.
    "
  [ x ]
  (py/call-attr activations "linear"  x ))

(defn relu 
  "Rectified Linear Unit.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    # Arguments
        x: Input tensor.
        alpha: float. Slope of the negative part. Defaults to zero.
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.

    # Returns
        A tensor.
    "
  [x & {:keys [alpha max_value threshold]
                       :or {alpha 0.0 threshold 0.0}} ]
    (py/call-attr-kw activations "relu" [x] {:alpha alpha :max_value max_value :threshold threshold }))

(defn selu 
  "Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is \"large enough\" (see references for more information).

    # Arguments
        x: A tensor or variable to compute the activation function for.

    # Returns
       The scaled exponential unit activation: `scale * elu(x, alpha)`.

    # Note
        - To be used together with the initialization \"lecun_normal\".
        - To be used together with the dropout variant \"AlphaDropout\".

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    "
  [ x ]
  (py/call-attr activations "selu"  x ))

(defn serialize 
  ""
  [ activation ]
  (py/call-attr activations "serialize"  activation ))

(defn sigmoid 
  "Sigmoid activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The sigmoid activation: `1 / (1 + exp(-x))`.
    "
  [ x ]
  (py/call-attr activations "sigmoid"  x ))

(defn softmax 
  "Softmax activation function.

    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.

    # Returns
        Tensor, output of softmax transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    "
  [x & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw activations "softmax" [x] {:axis axis }))

(defn softplus 
  "Softplus activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The softplus activation: `log(exp(x) + 1)`.
    "
  [ x ]
  (py/call-attr activations "softplus"  x ))

(defn softsign 
  "Softsign activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The softsign activation: `x / (abs(x) + 1)`.
    "
  [ x ]
  (py/call-attr activations "softsign"  x ))

(defn tanh 
  "Hyperbolic tangent activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The hyperbolic activation:
        `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    "
  [ x ]
  (py/call-attr activations "tanh"  x ))
