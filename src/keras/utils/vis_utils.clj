(ns keras.utils.vis-utils
  "Utilities related to model visualization."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce vis-utils (import-module "keras.utils.vis_utils"))

(defn model-to-dot 
  "Convert a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    "
  [ & {:keys [model show_shapes show_layer_names rankdir]
       :or {show_shapes false show_layer_names true rankdir "TB"}} ]
  
   (py/call-attr-kw vis-utils "model_to_dot" [] {:model model :show_shapes show_shapes :show_layer_names show_layer_names :rankdir rankdir }))

(defn plot-model 
  "Converts a Keras model to dot format and save to a file.

    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    "
  [ & {:keys [model to_file show_shapes show_layer_names rankdir]
       :or {to_file "model.png" show_shapes false show_layer_names true rankdir "TB"}} ]
  
   (py/call-attr-kw vis-utils "plot_model" [] {:model model :to_file to_file :show_shapes show_shapes :show_layer_names show_layer_names :rankdir rankdir }))
