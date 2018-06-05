# SSD 源码阅读笔记

## AnnotatedData layer解读

### 涉及的文件
* `ssd/src/caffe/layers/annotated_data_layer.cpp`
* `ssd/include/caffe/layers/annotated_data_layer.hpp`
* `ssd/src/caffe/proto/caffe.proto`
* `ssd/src/caffe/data_transformer.cpp`
* `ssd/src/caffe/util/sampler.cpp`
*  `ssd/src/caffe/util/im_transforms.cpp`
*  `ssd/src/caffe/util/bbox_util.cpp`

### DataLayerSetUp()函数

下面是整个`DataLayerSetUp()`的源代码

```cpp
template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}
```

接下来将上面代码一段一段进行分析

#### 001

```cpp
const int batch_size = this->layer_param_.data_param().batch_size();
```

该行代码主要获取`data_param`中的 `batch_size`值， `DataParameter`的具体定义(详见 `caffe.proto` )下，其值参见`train.prototxt` 的 `AnnotatedData Layer`。

```protobuf
message DataParameter {
  enum DB {
    LEVELDB = 0;
    LMDB = 1;
  }
  // Specify the data source.
  optional string source = 1;
  // Specify the batch size.
  optional uint32 batch_size = 4;
  // The rand_skip variable is for the data layer to skip a few data points
  // to avoid all asynchronous sgd clients to start at the same point. The skip
  // point would be set as rand_skip * rand(0,1). Note that rand_skip should not
  // be larger than the number of keys in the database.
  // DEPRECATED. Each solver accesses a different subset of the database.
  optional uint32 rand_skip = 7 [default = 0];
  optional DB backend = 8 [default = LEVELDB];
  // DEPRECATED. See TransformationParameter. For data pre-processing, we can do
  // simple scaling and subtracting the data mean, if provided. Note that the
  // mean subtraction is always carried out before scaling.
  optional float scale = 2 [default = 1];
  optional string mean_file = 3;
  // DEPRECATED. See TransformationParameter. Specify if we would like to randomly
  // crop an image.
  optional uint32 crop_size = 5 [default = 0];
  // DEPRECATED. See TransformationParameter. Specify if we want to randomly mirror
  // data.
  optional bool mirror = 6 [default = false];
  // Force the encoded image to have 3 color channels
  optional bool force_encoded_color = 9 [default = false];
  // Prefetch queue (Number of batches to prefetch to host memory, increase if
  // data access bandwidth varies).
  optional uint32 prefetch = 10 [default = 4];
}
```

#### 002

```cpp
 const AnnotatedDataParameter& anno_data_param =
     this->layer_param_.annotated_data_param();
 for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
   batch_samplers_.push_back(anno_data_param.batch_sampler(i));
 }
 label_map_file_ = anno_data_param.label_map_file();
```

这几行代码主要获取 `train.prototxt` 中 `AnnotatedData` 里的 `annotated_data_param`参数 ，并赋给指定的变量。

`batch_samplers_`是一个类型为`BatchSampler`的`vector`，其声明如下：

```cpp
// annotated_data_layer.hpp
vector<BatchSampler> batch_samplers_;
```

`AnnotatedDataParameter`的定义（`caffe.proto`）如下：

```protobuf
message AnnotatedDataParameter {
  // Define the sampler.
  repeated BatchSampler batch_sampler = 1;
  // Store label name and label id in LabelMap format.
  optional string label_map_file = 2;
  // If provided, it will replace the AnnotationType stored in each
  // AnnotatedDatum.
  optional AnnotatedDatum.AnnotationType anno_type = 3;
}
```

上面参数定义中又包括其他类型，其定义如下

```protobuf
// Sample a batch of bboxes with provided constraints.
message BatchSampler {
  // Use original image as the source for sampling.
  optional bool use_original_image = 1 [default = true];

  // Constraints for sampling bbox.
  optional Sampler sampler = 2;

  // Constraints for determining if a sampled bbox is positive or negative.
  optional SampleConstraint sample_constraint = 3;

  // If provided, break when found certain number of samples satisfing the
  // sample_constraint.
  optional uint32 max_sample = 4;

  // Maximum number of trials for sampling to avoid infinite loop.
  optional uint32 max_trials = 5 [default = 100];
}
```

```protobuf
// Sample a bbox in the normalized space [0, 1] with provided constraints.
message Sampler {
  // Minimum scale of the sampled bbox.
  optional float min_scale = 1 [default = 1.];
  // Maximum scale of the sampled bbox.
  optional float max_scale = 2 [default = 1.];

  // Minimum aspect ratio of the sampled bbox.
  optional float min_aspect_ratio = 3 [default = 1.];
  // Maximum aspect ratio of the sampled bbox.
  optional float max_aspect_ratio = 4 [default = 1.];
}
```

```protobuf
// Constraints for selecting sampled bbox.
message SampleConstraint {
  // Minimum Jaccard overlap between sampled bbox and all bboxes in
  // AnnotationGroup.
  optional float min_jaccard_overlap = 1;
  // Maximum Jaccard overlap between sampled bbox and all bboxes in
  // AnnotationGroup.
  optional float max_jaccard_overlap = 2;

  // Minimum coverage of sampled bbox by all bboxes in AnnotationGroup.
  optional float min_sample_coverage = 3;
  // Maximum coverage of sampled bbox by all bboxes in AnnotationGroup.
  optional float max_sample_coverage = 4;

  // Minimum coverage of all bboxes in AnnotationGroup by sampled bbox.
  optional float min_object_coverage = 5;
  // Maximum coverage of all bboxes in AnnotationGroup by sampled bbox.
  optional float max_object_coverage = 6;
}
```

```protobuf
// An extension of Datum which contains "rich" annotations.
message AnnotatedDatum {
  enum AnnotationType {
    BBOX = 0;
  }
  optional Datum datum = 1;
  // If there are "rich" annotations, specify the type of annotation.
  // Currently it only supports bounding box.
  // If there are no "rich" annotations, use label in datum instead.
  optional AnnotationType type = 2;
  // Each group contains annotation for a particular class.
  repeated AnnotationGroup annotation_group = 3;
}
```

```protobuf
message Datum {
  optional int32 channels = 1;
  optional int32 height = 2;
  optional int32 width = 3;
  // the actual image data, in bytes
  optional bytes data = 4;
  optional int32 label = 5;
  // Optionally, the datum could also hold float data.
  repeated float float_data = 6;
  // If true data contains an encoded image that need to be decoded
  optional bool encoded = 7 [default = false];
}
```


```protobuf
// Group of annotations for a particular label.
message AnnotationGroup {
  optional int32 group_label = 1;
  repeated Annotation annotation = 2;
}
```

```protobuf
// Annotation for each object instance.
message Annotation {
  optional int32 instance_id = 1 [default = 0];
  optional NormalizedBBox bbox = 2;
}
```

```protobuf
// The normalized bounding box [0, 1] w.r.t. the input image size.
message NormalizedBBox {
  optional float xmin = 1;
  optional float ymin = 2;
  optional float xmax = 3;
  optional float ymax = 4;
  optional int32 label = 5;
  optional bool difficult = 6;
  optional float score = 7;
  optional float size = 8;
}
```

#### 003

```cpp
// Make sure dimension is consistent within batch.
const TransformationParameter& transform_param =
  this->layer_param_.transform_param();
if (transform_param.has_resize_param()) {
  if (transform_param.resize_param().resize_mode() ==
      ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
    CHECK_EQ(batch_size, 1)
      << "Only support batch size of 1 for FIT_SMALL_SIZE.";
  }
}
```

这段代码主要获取 `transform_param`参数，`TransformationParamrter`类型及相关类型定义如下：

```protobuf
// Message that stores parameters used to apply transformation
// to the data layer's data
message TransformationParameter {
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  optional float scale = 1 [default = 1];
  // Specify if we want to randomly mirror data.
  optional bool mirror = 2 [default = false];
  // Specify if we would like to randomly crop an image.
  optional uint32 crop_size = 3 [default = 0];
  optional uint32 crop_h = 11 [default = 0];
  optional uint32 crop_w = 12 [default = 0];

  // mean_file and mean_value cannot be specified at the same time
  optional string mean_file = 4;
  // if specified can be repeated once (would substract it from all the channels)
  // or can be repeated the same number of times as channels
  // (would subtract them from the corresponding channel)
  repeated float mean_value = 5;
  // Force the decoded image to have 3 color channels.
  optional bool force_color = 6 [default = false];
  // Force the decoded image to have 1 color channels.
  optional bool force_gray = 7 [default = false];
  // Resize policy
  optional ResizeParameter resize_param = 8;
  // Noise policy
  optional NoiseParameter noise_param = 9;
  // Distortion policy
  optional DistortionParameter distort_param = 13;
  // Expand policy
  optional ExpansionParameter expand_param = 14;
  // Constraint for emitting the annotation after transformation.
  optional EmitConstraint emit_constraint = 10;
}
```

```protobuf
// Message that stores parameters used by data transformer for resize policy
message ResizeParameter {
  //Probability of using this resize policy
  optional float prob = 1 [default = 1];

  enum Resize_mode {
    WARP = 1;
    FIT_SMALL_SIZE = 2;
    FIT_LARGE_SIZE_AND_PAD = 3;
  }
  optional Resize_mode resize_mode = 2 [default = WARP];
  optional uint32 height = 3 [default = 0];
  optional uint32 width = 4 [default = 0];
  // A parameter used to update bbox in FIT_SMALL_SIZE mode.
  optional uint32 height_scale = 8 [default = 0];
  optional uint32 width_scale = 9 [default = 0];

  enum Pad_mode {
    CONSTANT = 1;
    MIRRORED = 2;
    REPEAT_NEAREST = 3;
  }
  // Padding mode for BE_SMALL_SIZE_AND_PAD mode and object centering
  optional Pad_mode pad_mode = 5 [default = CONSTANT];
  // if specified can be repeated once (would fill all the channels)
  // or can be repeated the same number of times as channels
  // (would use it them to the corresponding channel)
  repeated float pad_value = 6;

  enum Interp_mode { //Same as in OpenCV
    LINEAR = 1;
    AREA = 2;
    NEAREST = 3;
    CUBIC = 4;
    LANCZOS4 = 5;
  }
  //interpolation for for resizing
  repeated Interp_mode interp_mode = 7;
}
```

```protobuf
// Message that stores parameters used by data transformer for transformation
// policy
message NoiseParameter {
  //Probability of using this resize policy
  optional float prob = 1 [default = 0];
  // Histogram equalized
  optional bool hist_eq = 2 [default = false];
  // Color inversion
  optional bool inverse = 3 [default = false];
  // Grayscale
  optional bool decolorize = 4 [default = false];
  // Gaussian blur
  optional bool gauss_blur = 5 [default = false];

  // JPEG compression quality (-1 = no compression)
  optional float jpeg = 6 [default = -1];

  // Posterization
  optional bool posterize = 7 [default = false];

  // Erosion
  optional bool erode = 8 [default = false];

  // Salt-and-pepper noise
  optional bool saltpepper = 9 [default = false];

  optional SaltPepperParameter saltpepper_param = 10;

  // Local histogram equalization
  optional bool clahe = 11 [default = false];

  // Color space conversion
  optional bool convert_to_hsv = 12 [default = false];

  // Color space conversion
  optional bool convert_to_lab = 13 [default = false];
}
```

```protobuf
// Message that stores parameters used by data transformer for distortion policy
message DistortionParameter {
  // The probability of adjusting brightness.
  optional float brightness_prob = 1 [default = 0.0];
  // Amount to add to the pixel values within [-delta, delta].
  // The possible value is within [0, 255]. Recommend 32.
  optional float brightness_delta = 2 [default = 0.0];

  // The probability of adjusting contrast.
  optional float contrast_prob = 3 [default = 0.0];
  // Lower bound for random contrast factor. Recommend 0.5.
  optional float contrast_lower = 4 [default = 0.0];
  // Upper bound for random contrast factor. Recommend 1.5.
  optional float contrast_upper = 5 [default = 0.0];

  // The probability of adjusting hue.
  optional float hue_prob = 6 [default = 0.0];
  // Amount to add to the hue channel within [-delta, delta].
  // The possible value is within [0, 180]. Recommend 36.
  optional float hue_delta = 7 [default = 0.0];

  // The probability of adjusting saturation.
  optional float saturation_prob = 8 [default = 0.0];
  // Lower bound for the random saturation factor. Recommend 0.5.
  optional float saturation_lower = 9 [default = 0.0];
  // Upper bound for the random saturation factor. Recommend 1.5.
  optional float saturation_upper = 10 [default = 0.0];

  // The probability of randomly order the image channels.
  optional float random_order_prob = 11 [default = 0.0];
}
```

```protobuf
// Message that stores parameters used by data transformer for expansion policy
message ExpansionParameter {
  //Probability of using this expansion policy
  optional float prob = 1 [default = 1];

  // The ratio to expand the image.
  optional float max_expand_ratio = 2 [default = 1.];
}
```

#### 004

```cpp
// Read a data point, and use it to initialize the top blob.
AnnotatedDatum& anno_datum = *(reader_.full().peek());

// Use data_transformer to infer the expected blob shape from anno_datum.
vector<int> top_shape =
    this->data_transformer_->InferBlobShape(anno_datum.datum());
this->transformed_data_.Reshape(top_shape);
// Reshape top[0] and prefetch_data according to the batch_size.
top_shape[0] = batch_size;
top[0]->Reshape(top_shape);
for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
  this->prefetch_[i].data_.Reshape(top_shape);
}
LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
```

`InferBlobShape(const Datum& datum)`函数

```
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }

  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), datum_width, datum_height,
                 &datum_width, &datum_height);
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h)? crop_h: datum_height;
  shape[3] = (crop_w)? crop_w: datum_width;
  return shape;
}
```

`InferBlobShape(const cv::Mat& cv_img)`函数

```cpp
#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
                 &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h: img_height;
  shape[3] = (crop_w)? crop_w: img_width;
  return shape;
}
```

`InferNewSize`函数

```cpp
void InferNewSize(const ResizeParameter& resize_param,
                  const int old_width, const int old_height,
                  int* new_width, int* new_height) {
  int height = resize_param.height();
  int width = resize_param.width();
  float orig_aspect = static_cast<float>(old_width) / old_height;
  float aspect = static_cast<float>(width) / height;

  switch (resize_param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      if (orig_aspect < aspect) {
        height = static_cast<int>(width / orig_aspect);
      } else {
        width = static_cast<int>(orig_aspect * height);
      }
      break;
    default:
      LOG(FATAL) << "Unknown resize mode.";
  }
  *new_height = height;
  *new_width = width;
}
```

`PREFETCH_COUNT`的定义在`ssd/include/caffe/layers/base_data_layer.hpp`中，定义如下

```cpp
// Prefetches batches (asynchronously if to GPU memory)
 static const int PREFETCH_COUNT = 3;
```

因为`train.prototxt`中没有设置`crop_size`,`crop_h`和`crop_w`，因此最终`top_shape`为`[batch_size, img_channels, height, width]`，`height`和`width`为`resize_param`中的值。

#### 005

```cpp
// label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
```

首先`output_labels_`的定义如下（`ssd/src/caffe/layers/base_data_layer.cpp`），这里应该为`true`：

```cpp
if (top.size() == 1) {
  output_labels_ = false;
} else {
  output_labels_ = true;
}
```

然后`has_anno_type_`为`true`，在生成数据时，指定了`anno_type`。
