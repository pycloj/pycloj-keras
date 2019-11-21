
(ns keras.constraints.Constraint
  ""
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

(defn Constraint [  ]
  ""
  (py/call-attr constraints "Constraint"   ))

(defn get-config [ self ]
  ""
  (py/call-attr constraints "get_config"  self ))
