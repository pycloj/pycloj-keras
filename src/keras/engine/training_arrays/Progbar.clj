(ns keras.engine.training-arrays.Progbar
  "Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training-arrays (import-module "keras.engine.training_arrays"))

(defn Progbar 
  "Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    "
  [target & {:keys [width verbose interval stateful_metrics]
                       :or {width 30 verbose 1 interval 0.05}} ]
    (py/call-attr-kw training-arrays "Progbar" [target] {:width width :verbose verbose :interval interval :stateful_metrics stateful_metrics }))
(defn add 
  ""
  [self n  & {:keys [values]} ]
    (py/call-attr-kw self "add" [n] {:values values }))
(defn update 
  "Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        "
  [self current  & {:keys [values]} ]
    (py/call-attr-kw self "update" [current] {:values values }))
