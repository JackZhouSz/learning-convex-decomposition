# Learning Convex Decomposition via Feature Fields
**[Project](https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/)**
 
 Yuezhi Yang, Qixing Huang, Mikaela Angelina Uy*, Nicholas Sharp*
 
 CVPR 2026 (Oral)

 ## Overview
 This work proposes a new formulation to the long-standing problem of convex decomposition through learning feature fields, enabling the first feed-forward model for open-world convex decomposition. Our method produces high-quality decompositions of 3D shapes into a union of convex bodies, which are essential to accelerate collision detection in physical simulation, amongst many other applications. The key insight is to adopt a feature learning approach and learn a continuous feature field that can later be clustered to yield a good convex decomposition via our self-supervised, purely-geometric objective derived from the classical definition of convexity. Our formulation can be used for single shape optimization, but more importantly, feature prediction unlocks scalable, self-supervised learning on large datasets resulting in the first learned open-world for convex decomposition. Experiments show that our decompositions are higher-quality than alternatives and generalize across open-world objects as well as across representations to meshes, CAD models, and even Gaussian splats.

## Pretrained Model
TODO

 ## Environment Set-up
 TODO

 ## Example Run
 TODO

 ## Citation
```
@inproceedings{learningconvexdecomp2026,
      title={Learning Convex Decomposition via Feature Fields}, 
      author={Yuezhi Yang and and Qixing Huang and Mikaela Angelina Uy and Nicholas Sharp},
      booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2026}
}
```

## References  
We borrow code from the following repositories:  
- [OpenLRM](https://github.com/3DTopia/OpenLRM)  
- [PyTorch 3D UNet](https://github.com/wolny/pytorch-3dunet)  
- [PVCNN](https://github.com/mit-han-lab/pvcnn)  
- [GenSDF](https://github.com/princeton-computational-imaging/gensdf)  

Many thanks to the authors for sharing their code! Review the license terms of this project before use.

## License
This project will download and install additional third-party softwares. Note that these softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use.

This source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0). Attribution for bundled and adapted third-party code is provided in the root [NOTICE](./NOTICE) file.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for information on how to submit contributions, including the required Developer Certificate of Origin sign-off.