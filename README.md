# Causal Models
A model for creating and handling Structural Causal models. Additional benefits:

- Easy creation and handling of twin networks.
- Run approximate interventional/counterfactual inference on SCMs with [Pyro](https://github.com/pyro-ppl/pyro).
- Run exact inference in SCMs with [pgmpy](https://github.com/pgmpy/pgmpy).

See `demo.ipynb` for examples of usage.

This is still very much research code. It's expected to be buggy, but the basics work.
It's useful for playing around with SCMs in a native Python class, and especially for
approximate inference.

## Directory structure

```
- scm.py                  |  contains the CausalModel class, which handles and generates independent noise SCMs (binary or continuous).
- exact.py                |  contains the ExactCounterfactual class for performing and timing exact counterfactual inference with pgmpy.
- approximate.py          |  contains the ApproximateCounterfactual class for performing approximate counterfactual inference with Pyro.
- CustomVariableElimination.py |  slightly modifies the pgmpy ExactInference file to time elimination-based inference.
- experiments.py          |  contains the Experiment class which generates experiments: an SCM, node of interest, evidence, and intervention.
- demo.ipynb              |  a Jupyter notebook giving examples of basic functionality.
```

## Citation

```
@misc{babylon2019twin,
  title={Copy, paste, infer: {A} robust analysis of twin networks for counterfactual inference},
  author={Graham, Logan and Lee, Ciar{\'a}n M. and Perov, Yura},
  year={2019},
  note={Accepted to the CausalML workshop at NeurIPS 2019}
}
```

# Copyright and Licence # Causal Models
A model for creating and handling Structural Causal models. Additional benefits:

- Easy creation and handling of twin networks.
- Run approximate interventional/counterfactual inference on SCMs with [Pyro](https://github.com/pyro-ppl/pyro).
- Run exact inference in SCMs with [pgmpy](https://github.com/pgmpy/pgmpy).

See `demo.ipynb` for examples of usage.

This is still very much research code. It's expected to be buggy, but the basics work.
It's useful for playing around with SCMs in a native Python class, and especially for
approximate inference.

## Directory structure

```
- scm.py                  |  contains the CausalModel class, which handles and generates independent noise SCMs (binary or continuous).
- exact.py                |  contains the ExactCounterfactual class for performing and timing exact counterfactual inference with pgmpy.
- approximate.py          |  contains the ApproximateCounterfactual class for performing approximate counterfactual inference with Pyro.
- CustomVariableElimination.py |  slightly modifies the pgmpy ExactInference file to time elimination-based inference.
- experiments.py          |  contains the Experiment class which generates experiments: an SCM, node of interest, evidence, and intervention.
- demo.ipynb              |  a Jupyter notebook giving examples of basic functionality.
```

## Citation

```
@misc{babylon2019twin,
  title={Copy, paste, infer: {A} robust analysis of twin networks for counterfactual inference},
  author={Graham, Logan and Lee, Ciar{\'a}n M. and Perov, Yura},
  year={2019},
  note={Accepted to the CausalML workshop at NeurIPS 2019}
}
```

# Copyright and Licence

Copyright 2019-2020 Babylon Health (Babylon Partners Limited).

MIT Licence.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright 2019-2020 Babylon Health (Babylon Partners Limited).

MIT Licence.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.