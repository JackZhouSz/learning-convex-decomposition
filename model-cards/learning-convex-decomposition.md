# Learning Convex Decomposition via Feature Fields Overview

## Description:  
This work proposes a new formulation to the long-standing problem of convex decomposition through learning feature fields, enabling the first feed-forward model for open-world convex decomposition. Our method produces high-quality decompositions of 3D shapes into a union of convex bodies, which are essential to accelerate collision detection in physical simulation, amongst many other applications. The key insight is to adopt a feature learning approach and learn a continuous feature field that can later be clustered to yield a good convex decomposition via our self-supervised, purely-geometric objective derived from the classical definition of convexity. Our formulation can be used for single shape optimization, but more importantly, feature prediction unlocks scalable, self-supervised learning on large datasets resulting in the first learned open-world for convex decomposition. Experiments show that our decompositions are higher-quality than alternatives and generalize across open-world objects as well as across representations to meshes, CAD models, and even Gaussian splats.  
Learning Convex Decomposition via Feature Fields was developed by NVIDIA.  
_This model is for research and development only._  


### License/Terms of Use:
Apache-2.0:
### Deployment Geography:  
Global

### Use Case:  
This model is used for convex decomposition.  
  

### Release Date:
**HuggingFace:** 06/02/2026 via https://huggingface.co/mikaelaangel/learning-convex-decomp-ckpt  

  

## Reference(s):
[Learning Convex Decomposition via Feature Fields](https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/)  
[OpenLRM](https://github.com/3DTopia/OpenLRM)  
[PyTorch 3D UNet](https://github.com/wolny/pytorch-3dunet)  
[PVCNN (Point-Voxel CNN)](https://github.com/mit-han-lab/pvcnn)  
[GenSDF](https://github.com/princeton-computational-imaging/gensdf)  
[NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin)  

## Model Architecture:   
**Architecture Type:** Trasnformer, CNN   
**Network Architecture:** PVCNN  
**This model was developed based on Not Applicable (N/A).**  
**Number of model parameters:** 106M  

## Computational Load
**Cumulative Compute:** Not Applicable (N/A)  
**Estimated Energy and Emissions for Model Training:** Not Applicable (N/A)  
  
 
## Input:  
**Input Type(s):** Other: 3DModel  
**Input Format(s):** Other: obj  
**Input Parameters:** Three-Dimensional (3D)  
**Other Properties Related to Input:** Sample a point cloud from an input 3D model of any format  
  

## Output:  
**Output Type(s):** Embeddings, Other: Decomposed3DMesh  
**Output Format:** Tensor, Other: obj  
**Output Parameters:** N-Dimensional (ND), Three-Dimensional (3D)  
**Other Properties Related to Output:** The output would have an N-dimensional embedding where it can be clustered to result in a segmented 3D model  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** Transformers    
**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere  
**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  


## Model Version(s): 
learning-convex-decomposition-release  

We can follow the instructions on the github repo: https://gitlab-master.nvidia.com/sil/learning-convex-decomposition-release/  
 
## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:** Other: 3Dmodel  

**Other Training Data Size:** 3D data containing 340k shapes from Objaverse  
**Data Collection Method by dataset:** Automated  
**Labeling Method by dataset:** Automated  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We used the Objaverse dataset but curated a subset of these that contains more informative models. This subset was used in previous projects like PartField before as well.

### Testing Dataset:

**Data Collection Method by dataset:** Automated  
**Labeling Method by dataset:** Automated  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 61 3D models from the V-HACD data and 200 models from PartObjaverse-Tiny

### Evaluation Dataset:
**Benchmark Score:** Undisclosed

**Data Collection Method by dataset:** Automated  
**Labeling Method by dataset:** Automated  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We evaluated on the standard datasets for 3D convex decomposition.



## Inference:
**Acceleration Engine:** PyTorch
**Test Hardware:**  NVIDIA A100

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.  
  

**Generated by NVIDIA Model Card Generator Toolkit.**