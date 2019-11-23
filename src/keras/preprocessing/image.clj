(ns keras.preprocessing.image
  "Utilities for real-time data augmentation on image data.
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

(defn apply-affine-transform 
  "Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation

    # Returns
        The transformed version of the input.
    "
  [ & {:keys [x theta tx ty shear zx zy row_axis col_axis channel_axis fill_mode cval order]
       :or {theta 0 tx 0 ty 0 shear 0 zx 1 zy 1 row_axis 0 col_axis 1 channel_axis 2 fill_mode "nearest" cval 0.0 order 1}} ]
  
   (py/call-attr-kw image "apply_affine_transform" [] {:x x :theta theta :tx tx :ty ty :shear shear :zx zx :zy zy :row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :order order }))

(defn apply-brightness-shift 
  "Performs a brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    "
  [ & {:keys [x brightness]} ]
   (py/call-attr-kw image "apply_brightness_shift" [] {:x x :brightness brightness }))

(defn apply-channel-shift 
  "Performs a channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    "
  [ & {:keys [x intensity channel_axis]
       :or {channel_axis 0}} ]
  
   (py/call-attr-kw image "apply_channel_shift" [] {:x x :intensity intensity :channel_axis channel_axis }))

(defn array-to-img 
  "Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either \"channels_first\" or \"channels_last\".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
        dtype: Dtype to use.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    "
  [ & {:keys [x data_format scale dtype]
       :or {scale true}} ]
  
   (py/call-attr-kw image "array_to_img" [] {:x x :data_format data_format :scale scale :dtype dtype }))

(defn img-to-array 
  "Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either \"channels_first\" or \"channels_last\".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    "
  [ & {:keys [img data_format dtype]} ]
   (py/call-attr-kw image "img_to_array" [] {:img img :data_format data_format :dtype dtype }))

(defn load-img 
  "Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode=\"grayscale\"`.
        color_mode: One of \"grayscale\", \"rgb\", \"rgba\". Default: \"rgb\".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are \"nearest\", \"bilinear\", and \"bicubic\".
            If PIL version 1.1.3 or newer is installed, \"lanczos\" is also
            supported. If PIL version 3.4.0 or newer is installed, \"box\" and
            \"hamming\" are also supported. By default, \"nearest\" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    "
  [ & {:keys [path grayscale color_mode target_size interpolation]
       :or {grayscale false color_mode "rgb" interpolation "nearest"}} ]
  
   (py/call-attr-kw image "load_img" [] {:path path :grayscale grayscale :color_mode color_mode :target_size target_size :interpolation interpolation }))

(defn random-brightness 
  "Performs a random brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness_range: Tuple of floats; brightness range.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    "
  [ & {:keys [x brightness_range]} ]
   (py/call-attr-kw image "random_brightness" [] {:x x :brightness_range brightness_range }))

(defn random-channel-shift 
  "Performs a random channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.
    "
  [ & {:keys [x intensity_range channel_axis]
       :or {channel_axis 0}} ]
  
   (py/call-attr-kw image "random_channel_shift" [] {:x x :intensity_range intensity_range :channel_axis channel_axis }))

(defn random-rotation 
  "Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Rotated Numpy image tensor.
    "
  [ & {:keys [x rg row_axis col_axis channel_axis fill_mode cval interpolation_order]
       :or {row_axis 1 col_axis 2 channel_axis 0 fill_mode "nearest" cval 0.0 interpolation_order 1}} ]
  
   (py/call-attr-kw image "random_rotation" [] {:x x :rg rg :row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))

(defn random-shear 
  "Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Sheared Numpy image tensor.
    "
  [ & {:keys [x intensity row_axis col_axis channel_axis fill_mode cval interpolation_order]
       :or {row_axis 1 col_axis 2 channel_axis 0 fill_mode "nearest" cval 0.0 interpolation_order 1}} ]
  
   (py/call-attr-kw image "random_shear" [] {:x x :intensity intensity :row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))

(defn random-shift 
  "Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Shifted Numpy image tensor.
    "
  [ & {:keys [x wrg hrg row_axis col_axis channel_axis fill_mode cval interpolation_order]
       :or {row_axis 1 col_axis 2 channel_axis 0 fill_mode "nearest" cval 0.0 interpolation_order 1}} ]
  
   (py/call-attr-kw image "random_shift" [] {:x x :wrg wrg :hrg hrg :row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))

(defn random-zoom 
  "Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    "
  [ & {:keys [x zoom_range row_axis col_axis channel_axis fill_mode cval interpolation_order]
       :or {row_axis 1 col_axis 2 channel_axis 0 fill_mode "nearest" cval 0.0 interpolation_order 1}} ]
  
   (py/call-attr-kw image "random_zoom" [] {:x x :zoom_range zoom_range :row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))

(defn save-img 
  "Saves an image stored as a Numpy array to a path or file object.

    # Arguments
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format,
            either \"channels_first\" or \"channels_last\".
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    "
  [ & {:keys [path x data_format file_format scale]
       :or {scale true}} ]
  
   (py/call-attr-kw image "save_img" [] {:path path :x x :data_format data_format :file_format file_format :scale scale }))
