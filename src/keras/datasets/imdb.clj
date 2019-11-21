
(ns keras.datasets.imdb
  "IMDB sentiment classification dataset.
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
(defonce imdb (import-module "keras.datasets.imdb"))

(defn -remove-long-seq [ & {:keys [maxlen seq label]} ]
  "Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    "
   (py/call-attr-kw imdb "_remove_long_seq" [] {:maxlen maxlen :seq seq :label label }))

(defn get-file 
  "Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

    # Returns
        Path to the downloaded file
    "
  [ & {:keys [fname origin untar md5_hash file_hash cache_subdir hash_algorithm extract archive_format cache_dir]
       :or {untar false cache_subdir "datasets" hash_algorithm "auto" extract false archive_format "auto"}} ]
  
   (py/call-attr-kw imdb "get_file" [] {:fname fname :origin origin :untar untar :md5_hash md5_hash :file_hash file_hash :cache_subdir cache_subdir :hash_algorithm hash_algorithm :extract extract :archive_format archive_format :cache_dir cache_dir }))

(defn get-word-index 
  "Retrieves the dictionary mapping words to word indices.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    "
  [ & {:keys [path]
       :or {path "imdb_word_index.json"}} ]
  
   (py/call-attr-kw imdb "get_word_index" [] {:path path }))

(defn load-data 
  "Loads the IMDB dataset.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        maxlen: sequences longer than this will be filtered out.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case `maxlen` is so low
            that no input sequence could be kept.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    "
  [ & {:keys [path num_words skip_top maxlen seed start_char oov_char index_from]
       :or {path "imdb.npz" skip_top 0 seed 113 start_char 1 oov_char 2 index_from 3}} ]
  
   (py/call-attr-kw imdb "load_data" [] {:path path :num_words num_words :skip_top skip_top :maxlen maxlen :seed seed :start_char start_char :oov_char oov_char :index_from index_from }))
