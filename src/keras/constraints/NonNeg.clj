
(ns keras.constraints.NonNeg
  "Constrains the weights to be non-negative.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw
                     att-type-map
                     ->py-dict
                     ->py-list
                     ]
             :as py]
            [clojure.pprint :as pp]))

(py/initialize!)
(defonce constraints (import-module "keras.constraints"))

(defn NonNeg [  ]
  "Constrains the weights to be non-negative.
    "
  (py/call-attr constraints "NonNeg"   ))

(defn get-config [ self ]
  ""
  (py/call-attr constraints "get_config"  self ))
