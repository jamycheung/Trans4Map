# Trans4Map
**Trans4Map: Revisiting Holistic Top-down Mapping from Egocentric Images to Allocentric Semantics with Vision Transformers**

![trans4map](fig_trans4map.png)

### Introduction

In this work, we propose an end-to-end one-stage Transformer-based framework for Mapping, termed Trans4Map. Our egocentric-to-allocentric mapping process includes three steps: (1) the efficient transformer extracts the contextual features from a batch of egocentric images; (2) the proposed Bidirectional Allocentric Memory (BAM) module projects egocentric features into the allocentric memory; (3) the map decoder parses the accumulated memory and predicts the top-down semantic segmentation map.

More detailed can be found in our [arxiv](https://arxiv.org/pdf/2207.06205.pdf) paper.



### Usage 

The code and model will be made publicly available soon.



## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citations

If you are interested in this work, please cite the following work:

```text
@article{chen2022trans4map,
  title={Trans4Map: Revisiting Holistic Top-down Mapping from Egocentric Images to Allocentric Semantics with Vision Transformers},
  author={Chen, Chang and Zhang, Jiaming and Yang, Kailun and Peng, Kunyu and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2207.06205},
  year={2022}
}
```
