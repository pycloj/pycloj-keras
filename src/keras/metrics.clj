(ns keras.metrics
  "Built-in metrics.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce metrics (import-module "keras.metrics"))

(defn MAE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MAE"  y_true y_pred ))

(defn MAPE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MAPE"  y_true y_pred ))

(defn MSE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MSE"  y_true y_pred ))

(defn MSLE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MSLE"  y_true y_pred ))

(defn accuracy 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "accuracy"  y_true y_pred ))

(defn binary-accuracy 
  ""
  [y_true y_pred & {:keys [threshold]
                       :or {threshold 0.5}} ]
    (py/call-attr-kw metrics "binary_accuracy" [y_true y_pred] {:threshold threshold }))

(defn binary-crossentropy 
  ""
  [y_true y_pred & {:keys [from_logits label_smoothing]
                       :or {from_logits false label_smoothing 0}} ]
    (py/call-attr-kw metrics "binary_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))

(defn categorical-accuracy 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "categorical_accuracy"  y_true y_pred ))

(defn categorical-crossentropy 
  ""
  [y_true y_pred & {:keys [from_logits label_smoothing]
                       :or {from_logits false label_smoothing 0}} ]
    (py/call-attr-kw metrics "categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))

(defn categorical-hinge 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "categorical_hinge"  y_true y_pred ))

(defn clone-metric 
  "Returns a clone of the metric if stateful, otherwise returns it as is."
  [ metric ]
  (py/call-attr metrics "clone_metric"  metric ))

(defn clone-metrics 
  "Clones the given metric list/dict."
  [ metrics ]
  (py/call-attr metrics "clone_metrics"  metrics ))

(defn cosine 
  ""
  [y_true y_pred & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw metrics "cosine" [y_true y_pred] {:axis axis }))

(defn cosine-proximity 
  ""
  [y_true y_pred & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw metrics "cosine_proximity" [y_true y_pred] {:axis axis }))

(defn cosine-similarity 
  ""
  [y_true y_pred & {:keys [axis]
                       :or {axis -1}} ]
    (py/call-attr-kw metrics "cosine_similarity" [y_true y_pred] {:axis axis }))
(defn deserialize 
  ""
  [config  & {:keys [custom_objects]} ]
    (py/call-attr-kw metrics "deserialize" [config] {:custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [identifier & {:keys [module_objects custom_objects printable_module_name]
                       :or {printable_module_name "object"}} ]
    (py/call-attr-kw metrics "deserialize_keras_object" [identifier] {:module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  ""
  [ identifier ]
  (py/call-attr metrics "get"  identifier ))

(defn hinge 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "hinge"  y_true y_pred ))

(defn kullback-leibler-divergence 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "kullback_leibler_divergence"  y_true y_pred ))

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
  (py/call-attr metrics "logcosh"  y_true y_pred ))

(defn mae 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mae"  y_true y_pred ))

(defn mape 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mape"  y_true y_pred ))

(defn mean-absolute-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_absolute_error"  y_true y_pred ))

(defn mean-absolute-percentage-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_absolute_percentage_error"  y_true y_pred ))

(defn mean-squared-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_squared_error"  y_true y_pred ))

(defn mean-squared-logarithmic-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_squared_logarithmic_error"  y_true y_pred ))

(defn mse 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mse"  y_true y_pred ))

(defn msle 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "msle"  y_true y_pred ))

(defn poisson 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "poisson"  y_true y_pred ))

(defn serialize 
  ""
  [ metric ]
  (py/call-attr metrics "serialize"  metric ))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr metrics "serialize_keras_object"  instance ))

(defn sparse-categorical-accuracy 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "sparse_categorical_accuracy"  y_true y_pred ))

(defn sparse-categorical-crossentropy 
  ""
  [y_true y_pred & {:keys [from_logits axis]
                       :or {from_logits false axis -1}} ]
    (py/call-attr-kw metrics "sparse_categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :axis axis }))

(defn sparse-top-k-categorical-accuracy 
  ""
  [y_true y_pred & {:keys [k]
                       :or {k 5}} ]
    (py/call-attr-kw metrics "sparse_top_k_categorical_accuracy" [y_true y_pred] {:k k }))

(defn squared-hinge 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "squared_hinge"  y_true y_pred ))

(defn top-k-categorical-accuracy 
  ""
  [y_true y_pred & {:keys [k]
                       :or {k 5}} ]
    (py/call-attr-kw metrics "top_k_categorical_accuracy" [y_true y_pred] {:k k }))
