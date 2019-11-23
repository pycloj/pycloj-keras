(ns keras.engine.input-layer.Node
  "A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer._inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer._outbound_nodes`.

    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    origin_node = inbound_layers[i]._inbound_nodes[node_indices[i]]
    input_tensors[i] == origin_node.output_tensors[tensor_indices[i]]

    A node from layer A to layer B is added to:
        A._outbound_nodes
        B._inbound_nodes
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce input-layer (import-module "keras.engine.input_layer"))

(defn Node 
  "A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer._inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer._outbound_nodes`.

    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    origin_node = inbound_layers[i]._inbound_nodes[node_indices[i]]
    input_tensors[i] == origin_node.output_tensors[tensor_indices[i]]

    A node from layer A to layer B is added to:
        A._outbound_nodes
        B._inbound_nodes
    "
  [ & {:keys [outbound_layer inbound_layers node_indices tensor_indices input_tensors output_tensors input_masks output_masks input_shapes output_shapes arguments]} ]
   (py/call-attr-kw input-layer "Node" [] {:outbound_layer outbound_layer :inbound_layers inbound_layers :node_indices node_indices :tensor_indices tensor_indices :input_tensors input_tensors :output_tensors output_tensors :input_masks input_masks :output_masks output_masks :input_shapes input_shapes :output_shapes output_shapes :arguments arguments }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr input-layer "get_config"  self ))
