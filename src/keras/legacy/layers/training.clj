
(ns keras.legacy.layers.Model
  (:require [keras.engine.training]))

(defonce Model keras.engine.training.Model/Model)


(defonce add-loss keras.engine.training.Model/add-loss)

(defonce add-update keras.engine.training.Model/add-update)

(defonce add-weight keras.engine.training.Model/add-weight)

(defonce assert-input-compatibility keras.engine.training.Model/assert-input-compatibility)

(defonce build keras.engine.training.Model/build)

(defonce built keras.engine.training.Model/built)

(defonce call keras.engine.training.Model/call)

(defonce compile keras.engine.training.Model/compile)

(defonce compute-mask keras.engine.training.Model/compute-mask)

(defonce compute-output-shape keras.engine.training.Model/compute-output-shape)

(defonce count-params keras.engine.training.Model/count-params)

(defonce evaluate keras.engine.training.Model/evaluate)

(defonce evaluate-generator keras.engine.training.Model/evaluate-generator)

(defonce fit keras.engine.training.Model/fit)

(defonce fit-generator keras.engine.training.Model/fit-generator)

(defonce get-config keras.engine.training.Model/get-config)

(defonce get-input-at keras.engine.training.Model/get-input-at)

(defonce get-input-mask-at keras.engine.training.Model/get-input-mask-at)

(defonce get-input-shape-at keras.engine.training.Model/get-input-shape-at)

(defonce get-layer keras.engine.training.Model/get-layer)

(defonce get-losses-for keras.engine.training.Model/get-losses-for)

(defonce get-output-at keras.engine.training.Model/get-output-at)

(defonce get-output-mask-at keras.engine.training.Model/get-output-mask-at)

(defonce get-output-shape-at keras.engine.training.Model/get-output-shape-at)

(defonce get-updates-for keras.engine.training.Model/get-updates-for)

(defonce get-weights keras.engine.training.Model/get-weights)

(defonce input keras.engine.training.Model/input)

(defonce input-mask keras.engine.training.Model/input-mask)

(defonce input-shape keras.engine.training.Model/input-shape)

(defonce input-spec keras.engine.training.Model/input-spec)

(defonce layers keras.engine.training.Model/layers)

(defonce load-weights keras.engine.training.Model/load-weights)

(defonce losses keras.engine.training.Model/losses)

(defonce non-trainable-weights keras.engine.training.Model/non-trainable-weights)

(defonce output keras.engine.training.Model/output)

(defonce output-mask keras.engine.training.Model/output-mask)

(defonce output-shape keras.engine.training.Model/output-shape)

(defonce predict keras.engine.training.Model/predict)

(defonce predict-generator keras.engine.training.Model/predict-generator)

(defonce predict-on-batch keras.engine.training.Model/predict-on-batch)

(defonce reset-states keras.engine.training.Model/reset-states)

(defonce run-internal-graph keras.engine.training.Model/run-internal-graph)

(defonce save keras.engine.training.Model/save)

(defonce save-weights keras.engine.training.Model/save-weights)

(defonce set-weights keras.engine.training.Model/set-weights)

(defonce state-updates keras.engine.training.Model/state-updates)

(defonce stateful keras.engine.training.Model/stateful)

(defonce summary keras.engine.training.Model/summary)

(defonce test-on-batch keras.engine.training.Model/test-on-batch)

(defonce to-json keras.engine.training.Model/to-json)

(defonce to-yaml keras.engine.training.Model/to-yaml)

(defonce train-on-batch keras.engine.training.Model/train-on-batch)

(defonce trainable-weights keras.engine.training.Model/trainable-weights)

(defonce updates keras.engine.training.Model/updates)

(defonce uses-learning-phase keras.engine.training.Model/uses-learning-phase)

(defonce weights keras.engine.training.Model/weights)
