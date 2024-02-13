# CSKD: Knowledge Distillation for Object Detectors via Cosine Similarity-Guided Features

# Requirements
* mmcv-full==1.4.7
* mmdet==2.22.0
* pycocotools==2.0.2
* yapf==0.40.1
* follow installation [MMDetection](https://mmdetection.readthedocs.io/en/stable/get_started.html) for more information

# Dataset
* Download the [MS-COCO](https://cocodataset.org/#home) dataset to ```mmdetection/data```

# Train

* single GPU
```
python tools/train.py configs/cskd/cskd_retina_r101_distill_r50_fpn_coco_2x.py
```

* multi GPU
```
bash tools/dist_train.sh --config=configs/cskd/cskd_retina_r101_distill_r50_fpn_coco_2x.py [gpu num]
```

# Test

* single GPU
```
python tools/test.py configs/cskd/cskd_retina_r101_distill_r50_fpn_coco_2x.py 
```

* multi GPU
```
bash tools/dist_train.sh --config=configs/cskd/cskd_retina_r101_distill_r50_fpn_coco_2x.py --checkpoint=cskd_retina_r101_distill_r50_fpn_coco_2x.pth [gpu num]
```

# Results
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Detector</th>
    <th class="tg-c3ow" colspan="2">Backbone</th>
    <th class="tg-c3ow" colspan="3">mAP</th>
    <th class="tg-c3ow" rowspan="2">Config</th>
    <th class="tg-c3ow" rowspan="2">Model</th>
  </tr>
  <tr>
    <th class="tg-0pky">Teacher</th>
    <th class="tg-0pky">Student</th>
    <th class="tg-c3ow">Teacher</th>
    <th class="tg-c3ow">Student</th>
    <th class="tg-c3ow">Student w/ CSKD</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="2">Faster-RCNN</td>
    <td class="tg-0pky">ResNet-101</td>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">39.8</td>
    <td class="tg-c3ow">38.4</td>
    <td class="tg-c3ow">41.0(+2.6)</td>
    <td class="tg-y02l"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_faster_rcnn_r50_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1kTXFLXQf8I72DDlpsnnObG0RWn8oQfcT/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">ResNet-18</td>
    <td class="tg-c3ow">38.4</td>
    <td class="tg-c3ow">34.5</td>
    <td class="tg-c3ow">36.6(+2.1)</td>
    <td class="tg-y02l"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_faster_rcnn_r50_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1tCZkGCewu2DWdHuc-NcdPSquZLpJ8AST/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="3">RetinaNet</td>
    <td class="tg-0pky">ResNet-101</td>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">38.9</td>
    <td class="tg-c3ow">37.4</td>
    <td class="tg-c3ow">40.2(+2.8)</td>
    <td class="tg-0pky"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_retina_r101_distill_r50_fpn_coco_2x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1zMolp7o8QjS6B8DSEjJBprVAT7RRu4H3/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet-101</td>
    <td class="tg-c3ow">ResNet-18</td>
    <td class="tg-c3ow">38.9</td>
    <td class="tg-c3ow">31.7</td>
    <td class="tg-c3ow">35.8(+4.1)</td>
    <td class="tg-c3ow"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_retina_r101_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1vNKt7pJzTmI4tTQX9j-4INEL21MLNOPO/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">ResNet-18</td>
    <td class="tg-c3ow">37.4</td>
    <td class="tg-c3ow">31.7</td>
    <td class="tg-c3ow">35.2(+3.5)</td>
    <td class="tg-c3ow"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_retina_r50_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1-x-6pAXE1eMpoUAqdNoV_Z7p_v20S5MF/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">FCOS</td>
    <td class="tg-0pky">ResNet-101</td>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">42.6</td>
    <td class="tg-c3ow">40.9</td>
    <td class="tg-c3ow">42.0(+1.1)</td>
    <td class="tg-c3ow"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_fcos_r101_distill_r50_fpn_coco_2x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1Qo-RPogCUNESU4Vn5MrFdFzhCDoF9cnf/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">ResNet-18</td>
    <td class="tg-c3ow">40.9</td>
    <td class="tg-c3ow">32.5</td>
    <td class="tg-c3ow">35.9(+3.4)</td>
    <td class="tg-0pky"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_fcos_r50_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/17q6IxzsRKUr6Vk-fbEL9IyRUCuA-cKXj/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">GFL</td>
    <td class="tg-0pky">ResNet-101</td>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">44.9</td>
    <td class="tg-c3ow">40.2</td>
    <td class="tg-c3ow">44.5(+4.3)</td>
    <td class="tg-0pky"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_faster_rcnn_r50_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1kTXFLXQf8I72DDlpsnnObG0RWn8oQfcT/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet-50</td>
    <td class="tg-c3ow">ResNet-18</td>
    <td class="tg-c3ow">42.9</td>
    <td class="tg-c3ow">35.8</td>
    <td class="tg-c3ow">38.9(+3.1)</td>
    <td class="tg-0pky"><a href="https://github.com/paper-id-11094/CSKD/blob/master/configs/cskd/cskd_faster_rcnn_r50_distill_r18_fpn_coco_1x.py" target="_blank" rel="noopener noreferrer">config</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1tCZkGCewu2DWdHuc-NcdPSquZLpJ8AST/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
</tbody>
</table>