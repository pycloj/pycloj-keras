(ns keras.callbacks.TensorBoard
  "TensorBoard basic visualizations.

    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```

    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings]
            (https://www.tensorflow.org/programmers_guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce callbacks (import-module "keras.callbacks"))

(defn TensorBoard 
  "TensorBoard basic visualizations.

    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```

    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings]
            (https://www.tensorflow.org/programmers_guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    "
  [ & {:keys [log_dir histogram_freq batch_size write_graph write_grads write_images embeddings_freq embeddings_layer_names embeddings_metadata embeddings_data update_freq]
       :or {log_dir "./logs" histogram_freq 0 batch_size 32 write_graph true write_grads false write_images false embeddings_freq 0 update_freq "epoch"}} ]
  
   (py/call-attr-kw callbacks "TensorBoard" [] {:log_dir log_dir :histogram_freq histogram_freq :batch_size batch_size :write_graph write_graph :write_grads write_grads :write_images write_images :embeddings_freq embeddings_freq :embeddings_layer_names embeddings_layer_names :embeddings_metadata embeddings_metadata :embeddings_data embeddings_data :update_freq update_freq }))

(defn on-batch-begin 
  ""
  [self  & {:keys [batch logs]} ]
    (py/call-attr-kw callbacks "on_batch_begin" [self] {:batch batch :logs logs }))

(defn on-batch-end 
  ""
  [self  & {:keys [batch logs]} ]
    (py/call-attr-kw callbacks "on_batch_end" [self] {:batch batch :logs logs }))

(defn on-epoch-begin 
  ""
  [self  & {:keys [epoch logs]} ]
    (py/call-attr-kw callbacks "on_epoch_begin" [self] {:epoch epoch :logs logs }))

(defn on-epoch-end 
  ""
  [self  & {:keys [epoch logs]} ]
    (py/call-attr-kw callbacks "on_epoch_end" [self] {:epoch epoch :logs logs }))

(defn on-train-begin 
  ""
  [self  & {:keys [logs]} ]
    (py/call-attr-kw callbacks "on_train_begin" [self] {:logs logs }))

(defn on-train-end 
  ""
  [self  & {:keys [_]} ]
    (py/call-attr-kw callbacks "on_train_end" [self] {:_ _ }))

(defn set-model 
  ""
  [self  & {:keys [model]} ]
    (py/call-attr-kw callbacks "set_model" [self] {:model model }))

(defn set-params 
  ""
  [self  & {:keys [params]} ]
    (py/call-attr-kw callbacks "set_params" [self] {:params params }))
