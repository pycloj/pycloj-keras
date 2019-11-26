(ns keras.losses
  "Built-in loss functions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce losses (import-module "keras.losses"))

(defn KLD 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "KLD"  y_true y_pred ))

(defn MAE 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "MAE"  y_true y_pred ))

(defn MAPE 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "MAPE"  y_true y_pred ))

(defn MSE 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "MSE"  y_true y_pred ))

(defn MSLE 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "MSLE"  y_true y_pred ))

(defn binary-crossentropy 
  ""
  [y_true y_pred & {:keys [from_logits label_smoothing]
                       :or {from_logits false label_smoothing 0}} ]
    (py/call-attr-kw losses "binary_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))

(defn categorical-crossentropy 
  ""
  [y_true y_pred & {:keys [from_logits label_smoothing]
                       :or {from_logits false label_smoothing 0}} ]
    (py/call-attr-kw losses "categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))

(defn categorical-hinge 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "categorical_hinge"  y_true y_pred ))

(defn cosine 
  ""
  [y_true y_pred & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw losses "cosine" [y_true y_pred] {:axis axis }))

(defn cosine-proximity 
  ""
  [y_true y_pred & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw losses "cosine_proximity" [y_true y_pred] {:axis axis }))

(defn cosine-similarity 
  ""
  [y_true y_pred & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw losses "cosine_similarity" [y_true y_pred] {:axis axis }))
(defn deserialize 
  ""
  [name  & {:keys [custom_objects]} ]
    (py/call-attr-kw losses "deserialize" [name] {:custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw losses "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  "Get the `identifier` loss function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The loss function or None if `identifier` is None.

    # Raises
        ValueError if unknown identifier.
    "
  [ identifier ]
  (py/call-attr losses "get"  identifier ))

(defn hinge 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "hinge"  y_true y_pred ))

(defn huber-loss 
  ""
  [y_true y_pred & {:keys [delta]
                       :or {delta 1.0}} ]
    (py/call-attr-kw losses "huber_loss" [y_true y_pred] {:delta delta }))

(defn is-categorical-crossentropy 
  ""
  [ loss ]
  (py/call-attr losses "is_categorical_crossentropy"  loss ))

(defn kld 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "kld"  y_true y_pred ))

(defn kullback-leibler-divergence 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "kullback_leibler_divergence"  y_true y_pred ))

(defn logcosh 
  "Logarithm of the hyperbolic cosine of the prediction error.

    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    # Returns
        Tensor with one scalar loss entry per sample.
    "
  [ y_true y_pred ]
  (py/call-attr losses "logcosh"  y_true y_pred ))

(defn mae 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mae"  y_true y_pred ))

(defn mape 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mape"  y_true y_pred ))

(defn mean-absolute-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_absolute_error"  y_true y_pred ))

(defn mean-absolute-percentage-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_absolute_percentage_error"  y_true y_pred ))

(defn mean-squared-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_squared_error"  y_true y_pred ))

(defn mean-squared-logarithmic-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_squared_logarithmic_error"  y_true y_pred ))

(defn mse 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mse"  y_true y_pred ))

(defn msle 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "msle"  y_true y_pred ))

(defn poisson 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "poisson"  y_true y_pred ))

(defn serialize 
  ""
  [ loss ]
  (py/call-attr losses "serialize"  loss ))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr losses "serialize_keras_object"  instance ))

(defn sparse-categorical-crossentropy 
  ""
  [y_true y_pred & {:keys [from_logits axis]
                       :or {from_logits false axis -1}} ]
    (py/call-attr-kw losses "sparse_categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :axis axis }))

(defn squared-hinge 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "squared_hinge"  y_true y_pred ))
