(ns keras.legacy.interfaces
  "Interface converters for Keras 1 support in Keras 2.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce interfaces (import-module "keras.legacy.interfaces"))

(defn add-weight-args-preprocessing 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "add_weight_args_preprocessing" [] {:args args :kwargs kwargs }))

(defn batchnorm-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "batchnorm_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn conv1d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "conv1d_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn conv2d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "conv2d_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn conv3d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "conv3d_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn convlstm2d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "convlstm2d_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn deconv2d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "deconv2d_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn embedding-kwargs-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "embedding_kwargs_preprocessor" [] {:args args :kwargs kwargs }))

(defn generate-legacy-interface 
  ""
  [ & {:keys [allowed_positional_args conversions preprocessor value_conversions object_type]
       :or {object_type "class"}} ]
  
   (py/call-attr-kw interfaces "generate_legacy_interface" [] {:allowed_positional_args allowed_positional_args :conversions conversions :preprocessor preprocessor :value_conversions value_conversions :object_type object_type }))

(defn generator-methods-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "generator_methods_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn get-updates-arg-preprocessing 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "get_updates_arg_preprocessing" [] {:args args :kwargs kwargs }))

(defn legacy-add-weight-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_add_weight_support" [] {:func func }))

(defn legacy-batchnorm-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_batchnorm_support" [] {:func func }))

(defn legacy-conv1d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_conv1d_support" [] {:func func }))

(defn legacy-conv2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_conv2d_support" [] {:func func }))

(defn legacy-conv3d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_conv3d_support" [] {:func func }))

(defn legacy-convlstm2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_convlstm2d_support" [] {:func func }))

(defn legacy-cropping2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_cropping2d_support" [] {:func func }))

(defn legacy-cropping3d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_cropping3d_support" [] {:func func }))

(defn legacy-deconv2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_deconv2d_support" [] {:func func }))

(defn legacy-dense-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_dense_support" [] {:func func }))

(defn legacy-dropout-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_dropout_support" [] {:func func }))

(defn legacy-embedding-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_embedding_support" [] {:func func }))

(defn legacy-gaussiandropout-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_gaussiandropout_support" [] {:func func }))

(defn legacy-gaussiannoise-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_gaussiannoise_support" [] {:func func }))

(defn legacy-generator-methods-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_generator_methods_support" [] {:func func }))

(defn legacy-get-updates-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_get_updates_support" [] {:func func }))

(defn legacy-global-pooling-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_global_pooling_support" [] {:func func }))

(defn legacy-input-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_input_support" [] {:func func }))

(defn legacy-lambda-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_lambda_support" [] {:func func }))

(defn legacy-model-constructor-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_model_constructor_support" [] {:func func }))

(defn legacy-pooling1d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_pooling1d_support" [] {:func func }))

(defn legacy-pooling2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_pooling2d_support" [] {:func func }))

(defn legacy-pooling3d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_pooling3d_support" [] {:func func }))

(defn legacy-prelu-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_prelu_support" [] {:func func }))

(defn legacy-recurrent-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_recurrent_support" [] {:func func }))

(defn legacy-separable-conv2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_separable_conv2d_support" [] {:func func }))

(defn legacy-spatialdropout1d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_spatialdropout1d_support" [] {:func func }))

(defn legacy-spatialdropoutNd-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_spatialdropoutNd_support" [] {:func func }))

(defn legacy-upsampling1d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_upsampling1d_support" [] {:func func }))

(defn legacy-upsampling2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_upsampling2d_support" [] {:func func }))

(defn legacy-upsampling3d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_upsampling3d_support" [] {:func func }))

(defn legacy-zeropadding2d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_zeropadding2d_support" [] {:func func }))

(defn legacy-zeropadding3d-support 
  ""
  [ & {:keys [func]} ]
   (py/call-attr-kw interfaces "legacy_zeropadding3d_support" [] {:func func }))

(defn raise-duplicate-arg-error 
  ""
  [ & {:keys [old_arg new_arg]} ]
   (py/call-attr-kw interfaces "raise_duplicate_arg_error" [] {:old_arg old_arg :new_arg new_arg }))

(defn recurrent-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "recurrent_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn separable-conv2d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "separable_conv2d_args_preprocessor" [] {:args args :kwargs kwargs }))

(defn zeropadding2d-args-preprocessor 
  ""
  [ & {:keys [args kwargs]} ]
   (py/call-attr-kw interfaces "zeropadding2d_args_preprocessor" [] {:args args :kwargs kwargs }))
