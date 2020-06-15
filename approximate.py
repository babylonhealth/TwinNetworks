import time
import pyro
import torch
import networkx as nx
import pyro.distributions as dist

torch.set_default_tensor_type(torch.DoubleTensor)

class ApproximateCounterfactual():
    """A class for performing Pyro-based approximate inference with a  Structural Causal Model.
    Has the ability to handle and operate on Twin Networks.
    Currently implements only Importance Sampling, but would be simple to extend to other
    sampling-based methods and variational inference.
    
    Example:
        ```
        from scm import CausalModel
        scm = CausalModel(continuous=True)
        scm.create(method="backward", n_nodes=10, max_parents=5)
        app = ApproximatePyro()
        app.construct(scm)
        evidence = scm.sample_observable_dict(n_samples=1, n_vars=4)
        intervention = {"N5": 1.25}
        node_of_interest = "N9"
        posterior = app.counterfactual_query(node_of_interest, evidence, intervention, 1000, 1000)
        ```
    """

    def __init__(self, prior_mean=None, prior_variance=None, verbose=False):
        """
        Initialize an instance of the class.
        
        Args:
            verbose (bool): if True, prints out timings during the inference stage. 
        """
        self.verbose = verbose
        self._intv_nodes = []
        self._noi_nodes = []
        self.prior_mean = 0. if prior_mean is None else prior_mean
        self.prior_variance = 1. if prior_variance is None else prior_variance

    def construct(self, scm):
        """Parameterize the Approximate class via a specific SCM.
        
        Args:
            scm (CausalModel): a CausalModel instance.
        """
        self.continuous = scm.continuous
        if self.continuous:
            self.exog_fn = dist.Normal(torch.tensor([self.prior_mean]), self.prior_variance)
            self.invert_fn = self.continuous_invert
        else:
            self.exog_fn = dist.Bernoulli(torch.tensor([0.5]))
            self.invert_fn = self.binary_invert
        self.scm = scm
        self.G_inference = self.scm.G

    def model(self, evidence={}, noise=None):
        """A generative Pyro model function for a Structural Causal Model.

        Args:
            evidence (dict): a dictionary of {node_name: value} evidence data.
            noise: If you want to pass in a list of tuples of (node, sample) for samples from a noise posterior.
        Returns:
            model_dict: a sample from the model in {node_name: value} format.
            
        TODO: Make all endogenous nodes deterministic rather than delta variables
        """
        model_dict = {}
        if noise is not None:  # `noise` would not be None if you've already performed abduction.
            noise_sample = [z for z in zip(self.scm._get_exog_nodes(), noise.sample())]
            model_dict = {z[0]: pyro.sample(z[0], dist.Delta(z[1])) for z in noise_sample}  # sample from posterior.
        for node in nx.topological_sort(self.G_inference):
            parents = sorted(list(self.G_inference.predecessors(node)))
            if self.scm._is_exog(node, self.G_inference):
                if noise is None:  # only create a noise random variable if a noise posterior is not passed
                    model_dict[node] = pyro.sample(node, self.exog_fn)
            else:  # all remaining endogenous nodes are deterministic functions of their parents
                exog_parent = [pa for pa in parents if pa[0] == "U"][0]
                parent_values = [model_dict[n] for n in parents if not self.scm._is_exog(n, self.G_inference)]
                predicted_data = self._scm_function(node, parent_values, model_dict[exog_parent])
                if node in evidence:
                    model_dict[node] = pyro.sample(node, dist.Delta(predicted_data), obs=evidence[node])  # TODO: Choose
                    # model_dict[node] = self._assign_delta_node(node, predicted_data, obs=evidence[node])
                else:
                    model_dict[node] = pyro.sample(node, dist.Delta(predicted_data))  # TODO: Choose
                    # model_dict[node] = self._assign_delta_node(node, predicted_data)
        return model_dict

    def _assign_delta_node(self, node, value, obs=None):
        """Helper for assigning delta nodes. If node is intervened on, make pyro RV so pyro.do works. Else, float."""
        if node not in self._intv_nodes + self._noi_nodes:
            return value
        else:
            return pyro.sample(node, dist.Delta(value), obs=obs)

    def guide(self, evidence={}, noise=None):
        """A "smart" guide function for the SCM model above which propagates the information
        from a deterministic node being observed to the noise node, so that you don't end up with many rejected samples.
        This is slightly different from the model schema for the sake of sampling efficiency.

        Args:
            evidence (dict): a dictionary of {node_name: value} evidence data.
            noise (None): a useless parameter that exists because in Pyro, the guide fn has the same inputs as the model fn. 
        Returns:
            model_dict (dict): a sample from the guide in {node_name: value} format.
            
        TODO: Make all endogenous nodes deterministic rather than delta variables
        """
        guide_dict = {}
        # the order is a little complex. Any observed nodes have to go first, then the non-twin endog, then twin.
        for node in self._get_guide_order(evidence):
            exog_parent = [n for n in self.G_inference.predecessors(node) if self.scm._is_exog(n, self.G_inference)][0]
            endog_parents = sorted([n for n in self.G_inference.predecessors(node)
                                    if not self.scm._is_exog(n, self.G_inference)])
            if endog_parents:
                parent_values = [guide_dict[n] for n in endog_parents]
            else:
                parent_values = []
            if node not in evidence:
                if exog_parent not in guide_dict:  # if you haven't already sampled the exog_parent
                    guide_dict[exog_parent] = pyro.sample(exog_parent, self.exog_fn)
            else:
                if not endog_parents:  # if node only has an exogenous parent
                    if exog_parent not in guide_dict:
                        guide_dict[exog_parent] = pyro.sample(exog_parent, dist.Delta(evidence[node]))  # TODO: Choose
                        # guide_dict[exog_parent] = self._assign_delta_node(exog_parent, evidence[node])
                else:  # if a node has exog & endog parents
                    if exog_parent not in guide_dict:
                        predicted_val = self._scm_function(node, parent_values)
                        exog_val = self.invert_fn(evidence[node], predicted_val)
                        guide_dict[exog_parent] = pyro.sample(exog_parent, dist.Delta(exog_val))  # TODO: Choose
                        # guide_dict[exog_parent] = self._assign_delta_node(exog_parent, exog_val)
            val = self._scm_function(node, parent_values, guide_dict[exog_parent])
            guide_dict[node] = pyro.sample(node, dist.Delta(val))  # TODO: Choose
            # guide_dict[node] = self._assign_delta_node(node, val)
        return guide_dict

    def _get_guide_order(self, evidence):
        """A helper function that finds the correct model order if the twin graph is the graph of inference.
        
        Args:
            evidence (dict): a dictionary of {node_name: value} evidence data.
        """
        if self.G_inference.is_twin:
            return self.scm._weave_sort_endog(evidence)
        else:
            return self.scm._get_endog_nodes()

    def binary_invert(self, obs, pred):
        """The smart guide inversion function for recovering the exogenous value in binary SCMs.
        Write your own implementation based on how exogenous variables act in your SCM model.
        In the vanilla implementation, if pred != obs, the exogenous variable value must be 1 (it is active, or "flips")

        Args:
            obs (tensor): the observed evidence value tensor
            pred (tensor): the predicted value tensor
        """
        return (pred != obs).double()

    def continuous_invert(self, obs, pred):
        """The smart guide inversion function for recovering the exogenous value in cntinuous SCMs.
        Write your own implementation based on how exogenous variables act in your SCM model.
        In the vanilla implementation, noise is additive.
        Thus, it must make up the difference between the observed and predicted values.

        Args:
            obs (tensor): the observed evidence value tensor
            pred (tensor): the predicted value tensor
        """
        return obs - pred

    def _binary_exog_scm_function(self, val, flippers):
        """The binary SCM noise-flipping function.
        If flippers = 1, then flippers will flip val (i.e. 0=>1, 1=>0)
        Write your own implementation based on how you want noise to enter.

        Args:
            val (tensor): the predicted value
            flippers (tensor): the values of the flipper variables (sampled or inferred).
        """
        ## TODO: Move to SCM class
        val[flippers.byte()] = 1. - val[flippers.byte()]
        return val

    def _continuous_exog_scm_function(self, val, noise=None):
        """The continuous SCM additive noise function.
        Noise adds to the predicted value.
        Write your own implementation based on how you want noise to enter.

        Args:
            val (tensor): the predicted value
            noise (tensor or None): the noise value.
        """
        return val if noise is None else val + noise

    def _scm_function(self, node, parents, exog=None):
        """A handler for using each node's generating function to generate the value for that node.

        Args:
            node (str): the name of the node (used for indexing in self._deterministic_function(...))
            parents (list): a list of tensors of the values of the parents.
            exog (None or tensor): optional value if the only parent of `node` is its exogenous node.
        """
        ## TODO: Move to SCM class
        if not parents:
            return exog
        else:
            parents = torch.cat([pa.reshape(-1, 1) for pa in parents], dim=1)
            val = self._deterministic_function(node, parents, exog)
            return val.flatten()

    def _deterministic_function(self, node, parents, exog=None):
        """
        Calls the relevant deterministic functions and exerts the influence of the noise functin.
        
        Args:
            node (str): the name of the node (used for indexing in self._deterministic_function(...))
            parents (list): a list of tensors of the values of the parents.
            exog (None or tensor): optional value if the only parent of `node` is its exogenous node.
        """
        ## TODO: Move to SCM class
        if self.continuous:
            val = self.G_inference.nodes[node]['fn'](parents).flatten()
            val = self._continuous_exog_scm_function(val, exog)
        else:
            val = self._binary_fn(node, parents)
            val = self._binary_exog_scm_function(val, exog)
        return val

    def _binary_fn(self, node, parent_values):
        """The functional form for generating a predicted value, without the effect of noise.
        This is just a threshold-based classification function with linear form.

        Args:
            node (str): the name of the node
            parent_values (list):  a list of tensors of the values of `node`'s parents.
        """
        thetas = torch.from_numpy(self.G_inference.nodes[node]['parameters']).double()
        v = ((thetas * parent_values).sum(dim=-1) > 0.5).double()
        return v
    
    def get_posterior(self, nodes=None, evidence=None, n_samples=1000, custom_model=None, custom_guide=None, twin=False):
        """Run importance sampling for the defined model and guide to form a joint posterior.
        
        Args:
            nodes (list): list of nodes of the desired joint posterior
            evidence (dict): a dictionary of observed values formatted as {node_name: val}
            n_samples (int): the number of samples to take for inference.
            custom_model (fn): if desired, a custom model function.
            custom_guide (fn): if desired, a custom guide function.
            twin (bool): whether or not to run posterior inference on the twin network.
        """
        if evidence is not None:
            evidence = {d: torch.tensor([evidence[d]]).double() for d in evidence}
        self.G_inference = self.scm.G if not twin else self.scm.twin_G
        model = custom_model if custom_model is not None else self.model
        guide = custom_guide if custom_guide is not None else self.guide
        posterior = pyro.infer.Importance(model, guide, n_samples)
        posterior.run(evidence)
        posterior = pyro.infer.EmpiricalMarginal(posterior, sites=nodes)
        return posterior

    def abduction(self, evidence=None, n_samples=1000):
        """Run importance sampling for the above model and guide to form the joint posterior over *exog. variables*.
        
        Args:
            evidence (dict): a dictionary of observed values formatted as {node_name: val}
            n_samples (int): the number of samples to take for inference.
        """
        return self.get_posterior(nodes=self.scm._get_exog_nodes(), evidence=evidence, n_samples=n_samples)

    def intervention_prediction(self, node_of_interest, intervention, posterior, n_samples):
        """Given an exogenous posterior, sample then return the mean of the node of interest.
        
        Args:
            node_of_interest (str): the name of the noode of interest
            intervention (dict): a dictionary of {node_name: value} interventions
            evidence (dict): a dictionary of {node_name: value} evidence
            n_samples (int): the number of samples to take from the posterior.
            
        """
        intervention = {k: torch.tensor([intervention[k]]).double().flatten() for k in intervention}
        self._intv_nodes = [k for k in intervention]
        intervened_model = pyro.do(self.model, data=intervention)
        estimate = []
        for s in range(n_samples):
            estimate.append(intervened_model(noise=posterior)[node_of_interest])
        self._intv_nodes = []
        return estimate

    def counterfactual_query(self,
                             node_of_interest,
                             evidence,
                             intervention,
                             n_abduction_samples,
                             n_posterior_samples,
                             distribution=False):
        """Run the standard 3-step counterfactual inference procedure.
        
        Args:
            node_of_interest (str): the name of the noode of interest
            intervention (dict): a dictionary of {node_name: value} interventions
            evidence (dict): a dictionary of {node_name: value} evidence
            n_abduction_samples (int): the number of samples to take during importance sampling.
            n_posterior_samples (int): the number of samples to take from the posterior.
            distribution (bool): if True, return samples. If False, returns mean and sd. of distribution.
        """
        # Abduction step
        self.G_inference = self.scm.G_original
        t_abduction = time.time()
        if self.verbose:
            print("Performing Abduction... ", end="", flush=True)
        posterior = self.abduction(evidence, n_abduction_samples)
        t_abduction = time.time() - t_abduction
        if self.verbose:
            print("✓ ({}s)".format(round(t_abduction, 3)))

        # Prediction step
        t_prediction = time.time()
        if self.verbose:
            print("Performing Intervention and Prediction... ", end="", flush=True)
        samples = self.intervention_prediction(node_of_interest, intervention, posterior, n_posterior_samples)
        t_prediction = time.time() - t_prediction
        if self.verbose:
            print("✓ ({}s)".format(round(t_prediction, 3)))
        if not distribution:
            return torch.cat(samples).mean().numpy(), torch.cat(samples).std().numpy()
        else:
            return torch.cat(samples).numpy()

    def twin_query(self, node_of_interest, evidence, intervention, n_samples, distribution=False, merge=False):
        """Run the twin network counterfactual inference procedure.

        Args:
            node_of_interest (str): the name of the noode of interest
            intervention (dict): a dictionary of {node_name: value} interventions
            evidence (dict): a dictionary of {node_name: value} evidence
            n_samples (int): the number of samples to take during importance sampling and from the posterior.
            distribution (bool): if True, return samples. If False, returns mean and sd. of distribution.
        """
        if not self.scm.twin_exists:
            self.scm.create_twin_network()
        if merge:
            self.scm.merge_in_twin(node_of_interest, intervention)
        self.G_inference = self.scm.twin_G
        intervention = {"{}tn".format(k): torch.tensor([intervention[k]]).double().flatten() for k in intervention}
        node_of_interest = "{}tn".format(node_of_interest) if "tn" not in node_of_interest else node_of_interest
        self._intv_nodes = [k for k in intervention]
        self._noi_nodes = [node_of_interest]
        intervened_model = pyro.do(self.model, data=intervention)
        intervened_guide = pyro.do(self.guide, data=intervention)
        if self.verbose:
            print("Performing Twin Network inference... ", end="", flush=True)
        t_twin = time.time()
        posterior = self.get_posterior(self._noi_nodes,
                                       evidence,
                                       n_samples,
                                       custom_model=intervened_model,
                                       custom_guide=intervened_guide,
                                       twin=True)
        t_twin = time.time() - t_twin
        if self.verbose:
            print("✓ ({}s)".format(round(t_twin, 3)))
        self._intv_nodes = []
        self._noi_nodes = []
        samples = posterior.sample_n((n_samples))
        if not distribution:
            return samples.mean().numpy(), samples.std().numpy()
        else:
            return samples