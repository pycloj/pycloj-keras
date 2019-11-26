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
  [ args kwargs ]
  (py/call-attr interfaces "add_weight_args_preprocessing"  args kwargs ))

(defn batchnorm-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "batchnorm_args_preprocessor"  args kwargs ))

(defn conv1d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "conv1d_args_preprocessor"  args kwargs ))

(defn conv2d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "conv2d_args_preprocessor"  args kwargs ))

(defn conv3d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "conv3d_args_preprocessor"  args kwargs ))

(defn convlstm2d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "convlstm2d_args_preprocessor"  args kwargs ))

(defn deconv2d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "deconv2d_args_preprocessor"  args kwargs ))

(defn embedding-kwargs-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "embedding_kwargs_preprocessor"  args kwargs ))

(defn generate-legacy-interface 
  ""
  [ & {:keys [allowed_positional_args conversions preprocessor value_conversions object_type]
       :or {object_type "class"}} ]
  
   (py/call-attr-kw interfaces "generate_legacy_interface" [] {:allowed_positional_args allowed_positional_args :conversions conversions :preprocessor preprocessor :value_conversions value_conversions :object_type object_type }))

(defn generator-methods-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "generator_methods_args_preprocessor"  args kwargs ))

(defn get-updates-arg-preprocessing 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "get_updates_arg_preprocessing"  args kwargs ))

(defn legacy-add-weight-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_add_weight_support"  func ))

(defn legacy-batchnorm-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_batchnorm_support"  func ))

(defn legacy-conv1d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_conv1d_support"  func ))

(defn legacy-conv2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_conv2d_support"  func ))

(defn legacy-conv3d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_conv3d_support"  func ))

(defn legacy-convlstm2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_convlstm2d_support"  func ))

(defn legacy-cropping2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_cropping2d_support"  func ))

(defn legacy-cropping3d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_cropping3d_support"  func ))

(defn legacy-deconv2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_deconv2d_support"  func ))

(defn legacy-dense-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_dense_support"  func ))

(defn legacy-dropout-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_dropout_support"  func ))

(defn legacy-embedding-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_embedding_support"  func ))

(defn legacy-gaussiandropout-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_gaussiandropout_support"  func ))

(defn legacy-gaussiannoise-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_gaussiannoise_support"  func ))

(defn legacy-generator-methods-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_generator_methods_support"  func ))

(defn legacy-get-updates-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_get_updates_support"  func ))

(defn legacy-global-pooling-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_global_pooling_support"  func ))

(defn legacy-input-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_input_support"  func ))

(defn legacy-lambda-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_lambda_support"  func ))

(defn legacy-model-constructor-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_model_constructor_support"  func ))

(defn legacy-pooling1d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_pooling1d_support"  func ))

(defn legacy-pooling2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_pooling2d_support"  func ))

(defn legacy-pooling3d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_pooling3d_support"  func ))

(defn legacy-prelu-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_prelu_support"  func ))

(defn legacy-recurrent-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_recurrent_support"  func ))

(defn legacy-separable-conv2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_separable_conv2d_support"  func ))

(defn legacy-spatialdropout1d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_spatialdropout1d_support"  func ))

(defn legacy-spatialdropoutNd-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_spatialdropoutNd_support"  func ))

(defn legacy-upsampling1d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_upsampling1d_support"  func ))

(defn legacy-upsampling2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_upsampling2d_support"  func ))

(defn legacy-upsampling3d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_upsampling3d_support"  func ))

(defn legacy-zeropadding2d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_zeropadding2d_support"  func ))

(defn legacy-zeropadding3d-support 
  ""
  [ func ]
  (py/call-attr interfaces "legacy_zeropadding3d_support"  func ))

(defn raise-duplicate-arg-error 
  ""
  [ old_arg new_arg ]
  (py/call-attr interfaces "raise_duplicate_arg_error"  old_arg new_arg ))

(defn recurrent-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "recurrent_args_preprocessor"  args kwargs ))

(defn separable-conv2d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "separable_conv2d_args_preprocessor"  args kwargs ))

(defn zeropadding2d-args-preprocessor 
  ""
  [ args kwargs ]
  (py/call-attr interfaces "zeropadding2d_args_preprocessor"  args kwargs ))
