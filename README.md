# Coarse-to-Fine

#### Prerequisites

- use separate conda env: `conda create -n coarse2fine python=3.5`, then:

```sh
source activate coarse2fine
conda install pytorch=0.2.0 -c pytorch
pip install -r requirements.txt
```

- [download](https://drive.google.com/file/d/18oMNo4yC01gwMjHcfmE-_G5qE7X5SLYt/view?usp=sharing) data-model, and copy it to the root directory, then:

```sh
unzip acl18coarse2fine_data_model.zip
```

#### Reference

[Coarse-to-Fine Decoding for Neural Semantic Parsing](http://homepages.inf.ed.ac.uk/s1478528/acl18-coarse2fine.pdf)
```
@article{dong2018coarse,
  title={Coarse-to-fine decoding for neural semantic parsing},
  author={Dong, Li and Lapata, Mirella},
  journal={arXiv preprint arXiv:1805.04793},
  year={2018}
}
```
