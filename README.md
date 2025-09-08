# Awesome 3d Datasets

## 🌟 Overview

- [Awesome 3d Datasets](#awesome-3d-datasets)
  - [🌟 Overview](#-overview)
  - [📄 Citation](#-citation)
  - [📚 3D Datasets Summary](#-3d-datasets-summary)
    - [👤 Human](#-human)
    - [🎯 Object](#-object)
    - [🏙️ Scene](#️-scene)
    - [📊 Modalities of 3D datasets](#-modalities-of-3d-datasets)
  - [⚒️ Applications](#️-applications)
    - [🔄 3D Reconstruct](#-3d-reconstruct)
    - [✨ 3D Generation](#-3d-generation)
    - [🎬 Video Generation](#-video-generation)
    - [🌐 World Models](#-world-models)

## 📄 Citation

> Coming soon

## 📚 3D Datasets Summary

### 👤 Human

| Dataset                                       | Modality                                     | Year | Granularity | Tasks                                                        | Size | Site                                                 |
| --------------------------------------------- | -------------------------------------------- | ---- | ----------- | ------------------------------------------------------------ | ---- | ---------------------------------------------------- |
| [GigaHands](https://arxiv.org/abs/2412.04244) | mesh and text annotations                    | 2025 | Human Hand  | 3D bimanual hand                                             |      | [Github](https://github.com/Kristen-Z/GigaHands)     |
| [H3WB](https://arxiv.org/pdf/2211.15692)      | RGB, 2D+3D Whole-body Keypoints, Camera Pose | 2022 | Human Body  | 3D Pose Estimation                                           |      | [Github](https://github.com/wholebody3d/wholebody3d) |
| [FaceScape](https://arxiv.org/pdf/2003.13989) | FaceScape                                    | 2020 | Human Face  | Classification, Segmentation, Reconstruction, Completion, Recognition |      | [Github](https://github.com/zhuhao-nju/facescape)    |

### 🎯 Object

| Dataset                                                      | Modality                                                     | Year | Granularity | Tasks                                                        | Size | Site                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ----------- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| [Aria Synthetic Environments](https://arxiv.org/pdf/2403.13064) | RGB + LiDAR                                                  | 2024 | Scene       | Open-vocabulary Detection, LiDAR Region Merging, Long-tailed Object Detection |      | [Link](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_download_dataset?utm_source=chatgpt.com) |
| [WildRGB-D](https://arxiv.org/pdf/2401.12592)                | RGB-D, Instance Masks, Camera Pose, Point Cloud              | 2024 | Object      | View Synthesis, Pose Estimation, 6D Object Tracking, 3D Reconstruction |      | [Github](https://wildrgbd.github.io/)                        |
| [Objaverse](https://arxiv.org/pdf/2304.00501)                | 3D Mesh + Text                                               | 2023 | Object      | 3D Asset Collection, Annotation, Multimodal Learning         |      | [Github](https://github.com/allenai/objaverse-xl)            |
| [StrobeNet](https://arxiv.org/abs/2105.08016)                | Multiview RGB dataset for implicit 3D reconstruction         | 2021 | Object      | 3D Reconstruction                                            |      | [Link](https://dzhange.github.io/StrobeNet/)                 |
| [Amazon Berkeley Objects](https://arxiv.org/abs/2110.06199)  | Multi View, Camera Intrinsics & PBR Materials, 3D Mesh       | 2021 | Object      | 3D Reconstruction, Multi-view Retrieval, Material Estimation |      | [Website](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) |
| [Fusion 360 Gallery Dataset](https://www.research.autodesk.com/app/uploads/2023/03/Fusion_360_Gallery__A_Dataset_and_Environment_for_Programmatic_CAD_Construction_from_Human_Design_Sequences.pdf_recB1A7wJLthITzJo.pdf) | Parametric CAD (B-Rep) models + 2D sequences + 3D meshes + assembly/joint info | 2021 | Object      | 3D reconstruction, segmentation, assembly prediction, sequential modeling |      | [Github](https://github.com/AutodeskAILab/Fusion360GalleryDataset) |
| [CO3Dv2](https://arxiv.org/pdf/2109.00512)                   | Multi-view RGB, Camera Pose, Ground-truth 3D Point Cloud     | 2021 | Object      | Novel View Synthesis, Category-level 3D Reconstruction       |      | [Github](https://github.com/facebookresearch/co3d)           |
| [Habitat 2.0](https://arxiv.org/pdf/2106.14405)              | RGB, Depth, Semantic Segmentation                            | 2021 | Object      | Pick, Place, Navigate, Open, Close, Rearrange                |      | [Link](https://sites.google.com/view/habitat2)               |
| [A Large Dataset of Object Scans](https://vladlen.info/papers/3d-scan-dataset.pdf) | RGBD / Point Cloud                                           | 2020 | Object      | Object Scanning, 3D Reconstruction, Object Categorization    |      | [Github](https://github.com/isl-org/redwood-3dscan)          |
| [3D-FUTURE](https://arxiv.org/pdf/2009.09633.pdf)            | Furniture CAD with Textures                                  | 2020 | Object      | Navigation, Exploration, Interaction                         |      | [Github](https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox) |
| [ABC](https://openaccess.thecvf.com/content_CVPR_2019/papers/Koch_ABC_A_Big_CAD_Model_Dataset_for_Geometric_Deep_Learning_CVPR_2019_paper.pdf) | CAD                                                          | 2019 | Object      | Shape Analysis, Segmentation, Surface Fitting                |      | [Link](https://deep-geometry.github.io/abc-dataset)          |
| [BlendedMVS](https://arxiv.org/pdf/1911.10127)               | Multi-view                                                   | 2019 | Object      | Reconstruction, Alignment, Evaluation                        |      | [Github](https://github.com/YoYo000/BlendedMVS)              |
| [Thingi10K](https://arxiv.org/pdf/1605.04797)                | Triangle Mesh                                                | 2016 | Object      | Scene Understanding, Semantic Segmentation, Layout Prediction |      | [Github](https://github.com/Thingi10K/Thingi10K)             |
| [ShapeNet](https://arxiv.org/pdf/1512.03012.pdf)             | 3D Mesh + Semantic                                           | 2015 | Object      | Single-view Reconstruction, Multi-view Reconstruction        |      | [Link](https://shapenet.org/)                                |
| [ModelNet](https://arxiv.org/pdf/1406.5670.pdf)              | CAD Models                                                   | 2015 | Object      | Classification, Segmentation, Retrieval, Reconstruction      |      | [Link](http://modelnet.cs.princeton.edu/#)                   |
| [PASCAL3D+](https://arxiv.org/pdf/1511.05175)                | CAD                                                          | 2014 | Object      | Scene Understanding, Object Detection, Semantic Segmentation |      | [Link](https://cvgl.stanford.edu/resources.html)             |

### 🏙️ Scene

| Dataset                                                      | Modality                                                     | Year | Granularity   | Tasks                                                        | Size | Site                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ------------- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) | Gaussian splatting dataset for sparse-view reconstruction    | 2025 | Scene         | 3D scene understanding, controllable scene generation, embodied agent navigation |      | [HuggingFace](https://huggingface.co/datasets/spatialverse/InteriorGS) |
| [AnyHome](https://arxiv.org/abs/2312.06644)                  | Text-to-3D indoor scenes with structured mesh and layout     | 2024 | Scene         | Novel View Synthesis, NeRF Pretraining                       |      | [Website](https://dl3dv-10k.github.io/DL3DV-10K/)            |
| [Aria Digital Twin](https://arxiv.org/pdf/2306.06362)        | RGB + Depth + Audio                                          | 2023 | Scene         | 3D Question Answering, Spatial Reasoning, Scene Understanding |      | [Link](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset?utm_source=chatgpt.com) |
| [PointOdyssey](https://arxiv.org/pdf/2307.15055)             | 3D Scenes                                                    | 2023 | Scene         | 3D Generation, Multimodal Learning, Simulation               |      | [Link](https://pointodyssey.com/)                            |
| [DIVA-360](https://arxiv.org/abs/2307.16897)                 | 360 degree multi-view dataset for dynamic neural fields      | 2023 | Scene         | Novel View Synthesis, NeRF Pretraining                       |      | [Website](https://dl3dv-10k.github.io/DL3DV-10K/)            |
| [DL3DV-10K](https://arxiv.org/pdf/2312.16256)                | RGB video frames, Camera Pose, Scene Meta                    | 2023 | Scene         | Novel View Synthesis, NeRF Pretraining                       |      | [Website](https://dl3dv-10k.github.io/DL3DV-10K/)            |
| [Kubric](https://arxiv.org/pdf/2203.03570)                   | Multi-view RGB, Camera Pose, Semantic Segmentation, Semantic Point Cloud | 2022 | Indoor Scene  | Semantic Mapping, 2.5D Reconstruction, View-consistent Semantics |      | [Github](https://github.com/google-research/kubric)          |
| [HyperSim](https://arxiv.org/abs/2011.02523)                 | RGB + Depth + Pose + Segmentation + Material + Lighting + 3D Mesh | 2021 | Indoor Scene  | Multi-task Scene Understanding                               |      | [Github](https://github.com/apple/ml-hypersim)               |
| [Virtual KITTI](https://arxiv.org/pdf/2001.10773)            | Synthetic Video                                              | 2020 | Scene         | 6D Pose Estimation, Object Detection, Benchmarking           |      | [Link](https://datasetninja.com/virtual-kitti)               |
| [RELLIS-3D](https://arxiv.org/abs/2011.12954)                | RGB, LiDAR point cloud, Stereo, GPS/IMU, Camera+LiDAR Pose, Semantic Labels | 2020 | Outdoor Scene | 3D Semantic Segmentation, Sensor Fusion, Autonomous Navigation |      | [Website](https://www.unmannedlab.org/research/RELLIS-3D)    |
| [3D-FRONT](https://openaccess.thecvf.com/content/ICCV2021/papers/Fu_3D-FRONT_3D_Furnished_Rooms_With_layOuts_and_semaNTics_ICCV_2021_paper.pdf?utm_source=chatgpt.com) | Room Layout + Meshes                                         | 2020 | Scene         | Scene Understanding, Layout Analysis, Object Arrangement     |      | [HuggingFace](https://huggingface.co/datasets/huanngzh/3D-Front) |
| [Structured3D](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540494.pdf) | Photo-realistic + Annotations                                | 2020 | Scene         | Reconstruction, Segmentation, Object Detection               |      | [Link](https://structured3d-dataset.org/#download)           |
| [Mapillary](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470579.pdf) | Image + Depth Map                                            | 2020 | Scene         | Reconstruction, Semantics, Viewpoint Estimation              |      | [Link](https://www.mapillary.com/dataset/depth)              |
| [Replica](https://arxiv.org/pdf/1906.05797)                  | Dense Mesh + HDR Texture + Semantic/Instance Labels + Mirror/Glass | 2019 | Indoor Scene  | Scene Graph Generation, Object Detection, Relationship Modeling |      | [Github](https://github.com/facebookresearch/Replica-Dataset) |
| [RealEstate10K](https://arxiv.org/pdf/1805.09817)            | Camera Poses Corresponding to Frames                         | 2018 | Scene         | Part Segmentation, Hierarchical Labeling, Shape Understanding |      | [Link](https://google.github.io/realestate10k/)              |
| [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/paper.pdf) | RGBD, Camera Pose, Segment Labels                            | 2018 | Scene         | Multisensory Perception, Object Interaction, Representation Learning |      | [Link](https://www.cs.cornell.edu/projects/megadepth/)       |
| [DeepMVS](https://arxiv.org/pdf/1804.00650)                  | RGB images                                                   | 2018 | Scene         | CAD Alignment, 3D Matching, Pose Estimation                  |      | [Link](https://phuang17.github.io/DeepMVS/mvs-synth.html)    |
| [ScanNet](https://arxiv.org/pdf/1702.04405)                  | RGB-D                                                        | 2017 | Indoor Scene  | Feature Matching, Registration, 3D Reconstruction            |      | [Link](http://www.scan-net.org/)                             |
| [Semantic3D](https://isprs-annals.copernicus.org/articles/IV-1-W1/91/2017/isprs-annals-IV-1-W1-91-2017.pdf) | Point Cloud + Classification                                 | 2016 | Scene         | Point Cloud Classification, Semantic Segmentation            |      | [Link](http://www.semantic3d.net/)                           |
| [SceneNN / ObjectNN](https://www.saikit.org/static/projects/sceneNN/home/pdf/dataset_3dv16.pdf) | RGB-D Indoor Scenes                                          | 2016 | Indoor Scene  | Multi-view Fusion, 3D Reconstruction, Semantic Segmentation  |      | [Link](https://hkust-vgd.github.io/scenenn/)                 |

### 📊 Modalities of 3D datasets 

> ✅ indicates supported modality
> 
> 📝 *Modality includes available signals like RGB, Depth, Pose, Segmentation, Flow, Mesh, Action...*


| Dataset                        | RGB-D | Point Cloud | Mesh | Multi-view | Implicit Field |
|--------------------------------|-------|-------------|------|------------|----------------|
| GigaHands                      | ❌    | ❌          | ✅   | ✅         | ✅             |
| InteriorGS                     | ❌    | ❌          | ❌   | ✅         | ✅             |
| WildRGB-D                       | ✅    | ✅          | ❌   | ❌         | ❌             |
| Aria Synthetic Environments    | ❌    | ✅          | ❌   | ❌         | ❌             |
| AnyHome                        | ❌    | ❌          | ✅   | ❌         | ✅             |
| DL3DV-10K                      | ❌    | ❌          | ❌   | ❌         | ✅             |
| PointOdyssey                   | ❌    | ✅          | ❌   | ❌         | ❌             |
| Aria Digital Twin              | ✅    | ❌          | ❌   | ❌         | ❌             |
| Objaverse                      | ❌    | ❌          | ✅   | ❌         | ✅             |
| DIVA-360                       | ❌    | ❌          | ✅   | ✅         | ✅             |
| H3WB                           | ❌    | ❌          | ❌   | ❌         | ❌             |
| Kubric                         | ❌    | ✅          | ❌   | ✅         | ❌             |
| Amazon Berkeley Objects        | ❌    | ❌          | ✅   | ✅         | ❌             |
| Fusion 360 Gallery Dataset     | ❌    | ❌          | ✅   | ❌         | ❌             |
| CO3Dv2                         | ❌    | ✅          | ❌   | ✅         | ❌             |
| HyperSim                       | ✅    | ❌          | ✅   | ❌         | ❌             |
| Habitat 2.0                    | ✅    | ❌          | ❌   | ❌         | ❌             |
| StrobeNet                      | ❌    | ✅          | ❌   | ✅         | ✅             |
| Virtual KITTI                  | ✅    | ❌          | ❌   | ✅         | ❌             |
| RELLIS-3D                      | ❌    | ✅          | ❌   | ❌         | ❌             |
| FaceScape                      | ❌    | ❌          | ✅   | ❌         | ❌             |
| A Large Dataset of Object Scans| ✅    | ✅          | ❌   | ❌         | ❌             |
| 3D-FRONT                       | ❌    | ❌          | ✅   | ❌         | ❌             |
| 3D-FUTURE                      | ❌    | ❌          | ✅   | ❌         | ❌             |
| Structured3D                   | ✅    | ❌          | ❌   | ❌         | ❌             |
| Mapillary                      | ✅    | ❌          | ❌   | ❌         | ❌             |
| ABC                            | ❌    | ❌          | ✅   | ❌         | ✅             |
| BlendedMVS                     | ❌    | ❌          | ❌   | ✅         | ❌             |
| Replica                        | ❌    | ❌          | ✅   | ❌         | ❌             |
| RealEstate10K                  | ❌    | ❌          | ❌   | ✅         | ❌             |
| MegaDepth                      | ✅    | ❌          | ❌   | ✅         | ❌             |
| DeepMVS                        | ❌    | ❌          | ❌   | ✅         | ❌             |
| ScanNet                        | ✅    | ❌          | ❌   | ❌         | ❌             |
| Thingi10K                      | ❌    | ❌          | ✅   | ❌         | ✅             |
| Semantic3D                     | ❌    | ✅          | ❌   | ❌         | ❌             |
| SceneNN / ObjectNN             | ✅    | ❌          | ❌   | ❌         | ❌             |
| ShapeNet                       | ❌    | ❌          | ✅   | ❌         | ❌             |
| ModelNet                       | ❌    | ❌          | ✅   | ❌         | ❌             |
| PASCAL3D+                      | ❌    | ❌          | ✅   | ❌         | ❌             |

## ⚒️ Applications

### 🔄 3D Reconstruct

| Title | Year | Paper | Website | Code | HuggingFace
| :--- | :--- | :--- | :--- | :--- | :--
| DUSt3R: Geometric 3D Vision Made Easy | 2024 | [📄 Paper](https://arxiv.org/abs/2312.14132) | [🌍 Website](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/) | [💾 Code](https://github.com/naver/dust3r) | -
| VGGT: Visual Geometry Grounded Transformer | 2025 | [📄 Paper](https://arxiv.org/abs/2503.11651) | [🌍 Website](https://vgg-t.github.io/) | [💾 Code](https://github.com/facebookresearch/vggt) | [😊 HuggingFace](https://huggingface.co/spaces/facebook/vggt) |
| $\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning | 2025 | [📄 Paper](https://arxiv.org/abs/2507.13347) | [🌍 Website](https://yyfz.github.io/pi3/) | [💾 Code](https://github.com/yyfz/Pi3) | [😊 HuggingFace](https://huggingface.co/spaces/yyfz233/Pi3) |
| MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds | 2024 | [📄 Paper](https://arxiv.org/abs/2412.06974) | [🌍 Website](https://mv-dust3rp.github.io/) | [💾 Code](https://github.com/facebookresearch/mvdust3r) | - |
| MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details | 2025 | [📄 Paper](https://arxiv.org/abs/2507.02546) | - | - | - |
| MASt3R: Grounding Image Matching in 3D | 2024 | [📄 Paper](https://arxiv.org/abs/2406.09756) | [🌍 Website](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/) | [💾 Code](https://github.com/naver/mast3r) | [😊 HuggingFace](https://huggingface.co/spaces/naver/MASt3R)
| Mickey: Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences | 2024 | [📄 Paper](https://arxiv.org/abs/2404.06337) | [🌍 Website](https://nianticlabs.github.io/mickey/) | [💾 Code](https://github.com/nianticlabs/mickey) | -
| StreamVGGT: Streaming 4D Visual Geometry Transformer | 2025 | [📄 Paper](https://arxiv.org/abs/2507.11539) | [🌍 Website](https://wzzheng.net/StreamVGGT/) | [💾 Code](https://github.com/wzzheng/StreamVGGT) | [😊 HuggingFace](https://huggingface.co/spaces/lch01/StreamVGGT)
| MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second | 2025 | [📄 Paper](https://arxiv.org/abs/2507.10065) | [🌍 Website](https://chenguolin.github.io/projects/MoVieS/) | [💾 Code](https://github.com/chenguolin/MoVieS) | -

### ✨ 3D Generation

| Title | Year | Paper | Website | Code | HuggingFace
| :--- | :--- | :--- | :--- | :--- | :--
| DreamFusion: Text-to-3D using 2D Diffusion | 2022 | [📄 Paper](https://arxiv.org/abs/2209.14988) | [🌍 Website](https://dreamfusion3d.github.io/) | - | -
| Magic3D: High-Resolution Text-to-3D Content Creation | 2023 | [📄 Paper](https://arxiv.org/abs/2211.10440) | [🌍 Website](https://research.nvidia.com/labs/dir/magic3d/) | - | -
| DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation | 2024 | [📄 Paper](https://arxiv.org/abs/2309.16653) | [🌍 Website](https://dreamgaussian.github.io/) | [💾 Code](https://github.com/dreamgaussian/dreamgaussian) | [😊 HuggingFace](https://huggingface.co/spaces/jiawei011/dreamgaussian)
| DreamMesh: Jointly Manipulating and Texturing Triangle Meshes for Text-to-3D Generation | 2025 | [📄 Paper](https://arxiv.org/abs/2409.07454) | [🌍 Website](https://dreammesh.github.io/) | - | -
| Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data | 2025 | [📄 Paper](https://arxiv.org/abs/2503.21694) | [🌍 Website](https://theericma.github.io/TriplaneTurbo/) | [💾 Code](https://github.com/theEricMa/TriplaneTurbo) | [😊 HuggingFace](https://huggingface.co/spaces/ZhiyuanthePony/TriplaneTurbo)
| MVDream: Multi-view Diffusion for 3D Generation | 2024 | [📄 Paper](https://arxiv.org/abs/2308.16512) | [🌍 Website](https://mv-dream.github.io/) | [💾 Code](https://github.com/bytedance/MVDream) | - 
| Structured 3D Latents for Scalable and Versatile 3D Generation | 2025 | [📄 Paper](https://arxiv.org/abs/2412.01506) | [🌍 Website](https://microsoft.github.io/TRELLIS/) | [💾 Code](https://github.com/Microsoft/TRELLIS) | [😊 HuggingFace](https://huggingface.co/spaces/trellis-community/TRELLIS)
| 3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation | 2024 | [📄 Paper](https://arxiv.org/abs/2403.09439) | - | - | -

### 🎬 Video Generation

| Title | Year | Paper | Website | Code | HuggingFace
| :--- | :--- | :--- | :--- | :--- | :--
| CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer | 2024 | [📄 Paper](https://arxiv.org/abs/2408.06072) | [🌍 Website](https://yzy-thu.github.io/CogVideoX-demo/) | [💾 Code](https://github.com/zai-org/CogVideo) | [😊 HuggingFace](https://huggingface.co/spaces/zai-org/CogVideoX-5B-Space)
| Wan: Open and Advanced Large-Scale Video Generative Models | 2025 | [📄 Paper](https://arxiv.org/abs/2503.20314) | [🌍 Website](https://wan.video/) | [💾 Code](https://github.com/Wan-Video/Wan2.1)| [😊 HuggingFace](https://huggingface.co/Wan-AI)
| Lumiere: A Space-Time Diffusion Model for Video Generation | 2024 | [📄 Paper](https://arxiv.org/abs/2401.12945) | [🌍 Website](https://lumiere-video.github.io/) | [💾 Code](https://github.com/lumiere-video/lumiere-video.github.io) | -
| Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning | 2024 | [📄 Paper](https://arxiv.org/abs/2311.10709) | [🌍 Website](https://emu-video.metademolab.com/) | - | -
| Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets | 2023 | [📄 Paper](https://arxiv.org/abs/2311.15127) | [🌍 Website](https://sv4d20.github.io/) | [💾 Code](https://github.com/Stability-AI/generative-models) | [😊 HuggingFace](https://huggingface.co/stabilityai/sv4d2.0)
| 3D-Aware Video Generation | 2023 | [📄 Paper](https://arxiv.org/abs/2206.14797) | [🌍 Website](https://sherwinbahmani.github.io/3dvidgen/) | [💾 Code](https://github.com/sherwinbahmani/3dvideogeneration/) | -
| World-consistent Video Diffusion with Explicit 3D Modeling | 2024 | [📄 Paper](https://arxiv.org/abs/2412.01821) | [🌍 Website](https://zqh0253.github.io/wvd/) | - | -
| IM-Portrait: Learning 3D-aware Video Diffusion for Photorealistic Talking Heads from Monocular Videos | 2025 | [📄 Paper](https://arxiv.org/abs/2504.19165) | [🌍 Website](https://y-u-a-n-l-i.github.io/projects/IM-Portrait/) | - | -
| Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling | 2025 | [📄 Paper](https://arxiv.org/abs/2507.07982) | [🌍 Website](https://geometryforcing.github.io/) | - | - 
| Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals | 2025 | [📄 Paper](https://arxiv.org/abs/2505.19386) | [🌍 Website](https://force-prompting.github.io/) | [💾 Code](https://github.com/brown-palm/force-prompting) | -
| PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation | 2024 | [📄 Paper](https://arxiv.org/abs/2409.18964) | [🌍 Website](https://stevenlsw.github.io/physgen/) | [💾 Code](https://github.com/stevenlsw/physgen) | -
| Tora: Trajectory-oriented Diffusion Transformer for Video Generation | 2025 | [📄 Paper](https://arxiv.org/abs/2407.21705) | [🌍 Website](https://ali-videoai.github.io/tora_video/) | [💾 Code](https://github.com/alibaba/Tora) | [😊 HuggingFace](https://huggingface.co/Alibaba-Research-Intelligence-Computing/Tora)
| CamI2V: Camera-Controlled Image-to-Video Diffusion Model | 2024 | [📄 Paper](https://arxiv.org/abs/2410.15957) | [🌍 Website](https://zgctroy.github.io/CamI2V/) | [💾 Code](https://github.com/ZGCTroy/CamI2V) | [😊 HuggingFace](https://huggingface.co/MuteApo/CamI2V/tree/main)

### 🌐 World Models

| Title | Year | Paper | Website | Code | HuggingFace
| :--- | :--- | :--- | :--- | :--- | :--
| Learning to Simulate Complex Physics with Graph Networks | 2020 | [📄 Paper](https://arxiv.org/abs/2002.09405) | - | - |
| Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids | 2019 | [📄 Paper](https://arxiv.org/abs/1810.01566) | - | - |
| Learning Mesh-Based Simulation with Graph Networks | 2021 | [📄 Paper](https://arxiv.org/abs/2010.03409) | - | - |
| SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation | 2021 | [📄 Paper](https://arxiv.org/abs/2011.07215) | - | [💾 Code](https://github.com/Xingyu-Lin/softgym) |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | 2023 | [📄 Paper](https://arxiv.org/abs/2308.04079) | - | [💾 Code](https://github.com/graphdeco-inria/gaussian-splatting) |
| Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis | 2023 | [📄 Paper](https://arxiv.org/abs/2308.09713) | - | - |
| 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering | 2024 | [📄 Paper](https://arxiv.org/abs/2310.08528) | - | - |
| Gaussian Splatting SLAM | 2024 | [📄 Paper](https://arxiv.org/abs/2312.06741) | - | - |
| Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians | 2024 | [📄 Paper](https://arxiv.org/abs/2405.16544) | - | - |
| ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation | 2025 | [📄 Paper](https://arxiv.org/abs/2506.23126) | - | - |
