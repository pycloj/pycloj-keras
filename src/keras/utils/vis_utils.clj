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

(defn add-edge 
  ""
  [ dot src dst ]
  (py/call-attr vis-utils "add_edge"  dot src dst ))

(defn is-model 
  ""
  [ layer ]
  (py/call-attr vis-utils "is_model"  layer ))

(defn is-wrapped-model 
  ""
  [ layer ]
  (py/call-attr vis-utils "is_wrapped_model"  layer ))

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
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.
        subgraph: whether to return a pydot.Cluster instance.

    # Returns
        A `pydot.Dot` instance representing the Keras model or
        a `pydot.Cluster` instance representing nested model if
        `subgraph=True`.
    "
  [model & {:keys [show_shapes show_layer_names rankdir expand_nested dpi subgraph]
                       :or {show_shapes false show_layer_names true rankdir "TB" expand_nested false dpi 96 subgraph false}} ]
    (py/call-attr-kw vis-utils "model_to_dot" [model] {:show_shapes show_shapes :show_layer_names show_layer_names :rankdir rankdir :expand_nested expand_nested :dpi dpi :subgraph subgraph }))

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
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.

    # Returns
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    "
  [model & {:keys [to_file show_shapes show_layer_names rankdir expand_nested dpi]
                       :or {to_file "model.png" show_shapes false show_layer_names true rankdir "TB" expand_nested false dpi 96}} ]
    (py/call-attr-kw vis-utils "plot_model" [model] {:to_file to_file :show_shapes show_shapes :show_layer_names show_layer_names :rankdir rankdir :expand_nested expand_nested :dpi dpi }))
