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
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "MAE" [] {:y_true y_true :y_pred y_pred }))

(defn MAPE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "MAPE" [] {:y_true y_true :y_pred y_pred }))

(defn MSE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "MSE" [] {:y_true y_true :y_pred y_pred }))

(defn MSLE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "MSLE" [] {:y_true y_true :y_pred y_pred }))

(defn binary-accuracy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "binary_accuracy" [] {:y_true y_true :y_pred y_pred }))

(defn binary-crossentropy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "binary_crossentropy" [] {:y_true y_true :y_pred y_pred }))

(defn categorical-accuracy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "categorical_accuracy" [] {:y_true y_true :y_pred y_pred }))

(defn categorical-crossentropy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "categorical_crossentropy" [] {:y_true y_true :y_pred y_pred }))

(defn cosine 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "cosine" [] {:y_true y_true :y_pred y_pred }))

(defn cosine-proximity 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "cosine_proximity" [] {:y_true y_true :y_pred y_pred }))

(defn deserialize 
  ""
  [ & {:keys [config custom_objects]} ]
   (py/call-attr-kw metrics "deserialize" [] {:config config :custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [ & {:keys [identifier module_objects custom_objects printable_module_name]
       :or {printable_module_name "object"}} ]
  
   (py/call-attr-kw metrics "deserialize_keras_object" [] {:identifier identifier :module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  ""
  [ & {:keys [identifier]} ]
   (py/call-attr-kw metrics "get" [] {:identifier identifier }))

(defn hinge 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "hinge" [] {:y_true y_true :y_pred y_pred }))

(defn kullback-leibler-divergence 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "kullback_leibler_divergence" [] {:y_true y_true :y_pred y_pred }))

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
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "logcosh" [] {:y_true y_true :y_pred y_pred }))

(defn mae 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mae" [] {:y_true y_true :y_pred y_pred }))

(defn mape 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mape" [] {:y_true y_true :y_pred y_pred }))

(defn mean-absolute-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mean_absolute_error" [] {:y_true y_true :y_pred y_pred }))

(defn mean-absolute-percentage-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mean_absolute_percentage_error" [] {:y_true y_true :y_pred y_pred }))

(defn mean-squared-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mean_squared_error" [] {:y_true y_true :y_pred y_pred }))

(defn mean-squared-logarithmic-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mean_squared_logarithmic_error" [] {:y_true y_true :y_pred y_pred }))

(defn mse 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "mse" [] {:y_true y_true :y_pred y_pred }))

(defn msle 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "msle" [] {:y_true y_true :y_pred y_pred }))

(defn poisson 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "poisson" [] {:y_true y_true :y_pred y_pred }))

(defn serialize 
  ""
  [ & {:keys [metric]} ]
   (py/call-attr-kw metrics "serialize" [] {:metric metric }))

(defn serialize-keras-object 
  ""
  [ & {:keys [instance]} ]
   (py/call-attr-kw metrics "serialize_keras_object" [] {:instance instance }))

(defn sparse-categorical-accuracy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "sparse_categorical_accuracy" [] {:y_true y_true :y_pred y_pred }))

(defn sparse-categorical-crossentropy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "sparse_categorical_crossentropy" [] {:y_true y_true :y_pred y_pred }))

(defn sparse-top-k-categorical-accuracy 
  ""
  [ & {:keys [y_true y_pred k]
       :or {k 5}} ]
  
   (py/call-attr-kw metrics "sparse_top_k_categorical_accuracy" [] {:y_true y_true :y_pred y_pred :k k }))

(defn squared-hinge 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw metrics "squared_hinge" [] {:y_true y_true :y_pred y_pred }))

(defn top-k-categorical-accuracy 
  ""
  [ & {:keys [y_true y_pred k]
       :or {k 5}} ]
  
   (py/call-attr-kw metrics "top_k_categorical_accuracy" [] {:y_true y_true :y_pred y_pred :k k }))
