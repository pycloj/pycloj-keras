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
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "KLD" [] {:y_true y_true :y_pred y_pred }))

(defn MAE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "MAE" [] {:y_true y_true :y_pred y_pred }))

(defn MAPE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "MAPE" [] {:y_true y_true :y_pred y_pred }))

(defn MSE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "MSE" [] {:y_true y_true :y_pred y_pred }))

(defn MSLE 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "MSLE" [] {:y_true y_true :y_pred y_pred }))

(defn binary-crossentropy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "binary_crossentropy" [] {:y_true y_true :y_pred y_pred }))

(defn categorical-crossentropy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "categorical_crossentropy" [] {:y_true y_true :y_pred y_pred }))

(defn categorical-hinge 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "categorical_hinge" [] {:y_true y_true :y_pred y_pred }))

(defn cosine 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "cosine" [] {:y_true y_true :y_pred y_pred }))

(defn cosine-proximity 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "cosine_proximity" [] {:y_true y_true :y_pred y_pred }))

(defn deserialize 
  ""
  [ & {:keys [name custom_objects]} ]
   (py/call-attr-kw losses "deserialize" [] {:name name :custom_objects custom_objects }))

(defn deserialize-keras-object 
  ""
  [ & {:keys [identifier module_objects custom_objects printable_module_name]
       :or {printable_module_name "object"}} ]
  
   (py/call-attr-kw losses "deserialize_keras_object" [] {:identifier identifier :module_objects module_objects :custom_objects custom_objects :printable_module_name printable_module_name }))

(defn get 
  "Get the `identifier` loss function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The loss function or None if `identifier` is None.

    # Raises
        ValueError if unknown identifier.
    "
  [ & {:keys [identifier]} ]
   (py/call-attr-kw losses "get" [] {:identifier identifier }))

(defn hinge 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "hinge" [] {:y_true y_true :y_pred y_pred }))

(defn kld 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "kld" [] {:y_true y_true :y_pred y_pred }))

(defn kullback-leibler-divergence 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "kullback_leibler_divergence" [] {:y_true y_true :y_pred y_pred }))

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
   (py/call-attr-kw losses "logcosh" [] {:y_true y_true :y_pred y_pred }))

(defn mae 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mae" [] {:y_true y_true :y_pred y_pred }))

(defn mape 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mape" [] {:y_true y_true :y_pred y_pred }))

(defn mean-absolute-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mean_absolute_error" [] {:y_true y_true :y_pred y_pred }))

(defn mean-absolute-percentage-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mean_absolute_percentage_error" [] {:y_true y_true :y_pred y_pred }))

(defn mean-squared-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mean_squared_error" [] {:y_true y_true :y_pred y_pred }))

(defn mean-squared-logarithmic-error 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mean_squared_logarithmic_error" [] {:y_true y_true :y_pred y_pred }))

(defn mse 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "mse" [] {:y_true y_true :y_pred y_pred }))

(defn msle 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "msle" [] {:y_true y_true :y_pred y_pred }))

(defn poisson 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "poisson" [] {:y_true y_true :y_pred y_pred }))

(defn serialize 
  ""
  [ & {:keys [loss]} ]
   (py/call-attr-kw losses "serialize" [] {:loss loss }))

(defn serialize-keras-object 
  ""
  [ & {:keys [instance]} ]
   (py/call-attr-kw losses "serialize_keras_object" [] {:instance instance }))

(defn sparse-categorical-crossentropy 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "sparse_categorical_crossentropy" [] {:y_true y_true :y_pred y_pred }))

(defn squared-hinge 
  ""
  [ & {:keys [y_true y_pred]} ]
   (py/call-attr-kw losses "squared_hinge" [] {:y_true y_true :y_pred y_pred }))
