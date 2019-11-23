(ns keras.preprocessing.image.Iterator
  "Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "keras.preprocessing.image"))

(defn Iterator 
  "Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    "
  [ & {:keys [n batch_size shuffle seed]} ]
   (py/call-attr-kw image "Iterator" [] {:n n :batch_size batch_size :shuffle shuffle :seed seed }))

(defn next 
  "For python 2.x.

        # Returns
            The next batch.
        "
  [ self ]
  (py/call-attr image "next"  self ))

(defn on-epoch-end 
  ""
  [ self ]
  (py/call-attr image "on_epoch_end"  self ))

(defn reset 
  ""
  [ self ]
  (py/call-attr image "reset"  self ))
