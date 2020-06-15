import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
from CustomVariableElimination import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling.Sampling import BayesianModelSampling


class ExactCounterfactual(object):
    """
    A class for performing Exact counterfactual inference in both the Standard and Twin Network approaches.

    N.B.: For logging time, this relies on a custom edit of pgmpy.inference.ExactInference.VariableElimination,
    where the query also returns (as a second return) the time it takes to perform factor marginalization.
    """

    def __init__(self, verbose=False, merge=False):
        """
        Initialize the class.

        Args:
            verbose: whether or not to automatically print the Twin & standard inference times.
            merge: whether or not to perform node merging.
        """
        self.verbose = verbose
        self.merge = merge

    def construct(self, causal_model=None, G=None, df=None, n_samples=20000):
        """
        Init Args:
            twin_network: a TwinNetwork class.
            G: a networkx graph describing the dependency relationships.
            df: a dataframe of samples from that graph, used to construct the conditional probability tables.
        """
        if causal_model is None:
            assert G is not None and df is not None, "Must initialize G and df if no TwinNetwork passed."
            self.G = G
            self.df = df
        else:
            self.scm = causal_model
            self.G = causal_model.G.copy()
            samples = causal_model.sample(n_samples)
            self.df = pd.DataFrame(samples, columns=causal_model.ordering)
        self.model = None  # reset
        self.twin_model = None  # reset
        self.counterfactual_model = None  # reset
        self._compile_model()

    def _compile_model(self):
        """
        Makes a pgmpy model out of a networkx graph and parameterizes its CPD with CPTs estimated from a model.
        """
        self.model = BayesianModel(list(self.G.edges))
        self._construct_CPD()

    def create_twin_network(self, node_of_interest, observed, intervention):
        """
        Generate self.twin_model based on the current model, then merge nodes and eliminate nodes that are conditionally
        independent of the counterfactual node of interest.

        Args:
            node_of_interest: the node of interest to perform inference on.
            observed: a dictionary of {node: observed_value} to condition on.
            intervention: a dictionary of {node: intervention_value} to intervene on.
        """
        self.twin_model = self.model.copy()
        self.twin_model.add_nodes_from(["{}tn".format(n) for n in list(self.twin_model.nodes)
                                        if len(list(self.model.predecessors(n))) != 0])  # add all non-noise nodes
        self.twin_model.add_edges_from([("{}tn".format(pa), "{}tn".format(ch)) for pa, ch in list(self.model.edges)
                                        if len(list(self.model.predecessors(pa))) != 0]) # add all non-noise edges
        self.twin_model.add_edges_from([(pa, "{}tn".format(ch)) for pa, ch in list(self.model.edges)
                                        if len(list(self.model.predecessors(pa))) == 0]) #add all noise edges
        # merge nodes if merge flag is true
        if self.merge:
            self.merge_nodes(node_of_interest, intervention)

        # get appropriately ordered CPTs for new merged representation
        duplicate_cpts = []
        for node in self.twin_model.nodes:
            if node[-2:] == "tn":  # if in the twin network model
                node_parents = list(self.twin_model.predecessors(node))
                non_twin_parents = [pa.replace("tn", "") for pa in node_parents]
                cpt = TabularCPD(node,
                                 2,
                                 self.model.get_cpds(node[:-2]).reorder_parents(non_twin_parents),
                                 node_parents,
                                 len(node_parents)*[2])
                duplicate_cpts.append(cpt)
        self.twin_model.add_cpds(*duplicate_cpts)

        # make model efficient
        modified_intervention = {n + "tn": intervention[n] for n in intervention}  # modify for twin network syntax
        self.intervene(modified_intervention, twin=True)
        self._eliminate_conditionally_independent(node_of_interest, observed, intervention)

    def _construct_CPD(self, counterfactual=False, df=None):
        cpt_list = []
        if df is None:
            df = self.df
        for node in self.G.nodes:
            cpt_list.append(self._get_node_CPT(node, df))
        if counterfactual:
            self.counterfactual_model.add_cpds(*cpt_list)
        else:
            self.model.add_cpds(*cpt_list)
        self.df = None # erase df to make object pickleable, otherwise the object becomes unpicklable. (Important for parallel processing)

    def _get_node_CPT(self, node, df=None):
        parents = list(self.G.predecessors(node))
        if len(parents) == 0:  # if root node (latent)
            mu = df[node].mean()
            return TabularCPD(node, 2, values=[[1 - mu], [mu]])
        elif len(parents) > 0:
            mus = df.groupby(parents)[node].mean().reset_index()
            uniques = mus[parents].drop_duplicates()
            parent_combos = list(product(*[[0, 1] for _ in parents]))
            appends = []
            for combo in parent_combos:
                if not (uniques == np.array(combo)).all(1).any():  # if value not enumerated in sample
                    appends.append(list(combo) + [0.5])  # add an uninformative prior
            add_df = pd.DataFrame(appends, columns=parents + [node])
            mus = pd.concat((mus, add_df), axis=0)
            mus = mus.sort_values(by=parents)
            mus = mus[node].values
            cpt = np.vstack((1. - mus, mus))
            cpt = TabularCPD(node,
                             2,
                             values=cpt,
                             evidence=parents,
                             evidence_card=len(parents) * [2])
            return cpt

    def query(self, var, observed, counterfactual=False, twin=False):
        """
        Run an arbitrary query by Variable Elimination.

        What is the analytic cost of this? You have to do K noise queries in a graph with K endog nodes + K exog
        nodes in normal CFI. In twin network inference, you have to do 1 query in a graph with 2K endog nodes + K
        exog nodes.

        Args:
            var: variable of interest, i.e. P(Var | Observed)
            observed: a dictionary of {node_name: observed_value} to condition on.
            counterfactual: if true, uses the counterfactual model. (self.counterfactual_model)
            twin: if true, uses the twin network model. (self.twin_model)

        Returns:

        """
        if not isinstance(var, list):
            var = [var]
        if twin:
            # time_start = time.time()
            infer = VariableElimination(self.efficient_twin_model)
            result, time_elapsed = infer.query(var, evidence=observed, stopwatch=True)
            self.twin_inference_time = time_elapsed
        elif counterfactual:
            # time_start = time.time()
            infer = VariableElimination(self.counterfactual_model)
            result, time_elapsed = infer.query(var, evidence=observed, stopwatch=True)
            self.standard_inference_time = self.joint_inference_time + time_elapsed
        else:
            infer = VariableElimination(self.model)
            result, time_elapsed = infer.query(var, evidence=observed, stopwatch=True)
        return result, time_elapsed

    def intervene(self, intervention, counterfactual=False, twin=False):
        """
        Performs the intervention on the BN object by setting the CPT to be deterministic and removing parents.

        Args:
            intervention: a dictionary of {node_name: intervention_value} to intervene on.
        """
        cpt_list = []
        if counterfactual and not twin:
            model = self.counterfactual_model
        elif twin and not counterfactual:
            model = self.twin_model
        else:
            model = self.model
        for node in intervention:
            if node in model.nodes:
                # do-calculus graph surgery: remove edges from parents
                parent_edges = [(pa, node) for pa in model.predecessors(node)]
                model.remove_edges_from(parent_edges)
                model.remove_node("U{}".format(node))
                # set new deterministic CPT
                value = intervention[node]
                cpt = [[], []]
                cpt[value] = [1]
                cpt[int(not bool(value))] = [0]
                new_cpt = TabularCPD(node, 2, values=cpt)
                cpt_list.append(new_cpt)
        # override existing CPTs
        model.add_cpds(*cpt_list)

    def abduction(self, observed, n_samples=None):
        # infer latent joint and store the time it takes
        noise_nodes = [n for n in self.G.nodes if len(list(self.G.predecessors(n))) == 0]
        new_joint, time_elapsed = self.query(noise_nodes, observed)
        self.joint_inference_time = time_elapsed
        new_joint = new_joint.values.ravel()
        # sample from network with new latent distribution
        ## sample from joint
        dim = 2 ** len(noise_nodes)
        val_idx = np.arange(dim)
        # define number of samples
        if n_samples is None:  # be careful with this!
            n_samples = min([30 * 2 ** (len(list(self.G.nodes)) - len(noise_nodes)), 100000])
        noise_sample_idx = np.random.choice(val_idx, size=n_samples, p=new_joint)
        vals = np.array(list(product(*[[0, 1] for _ in range(len(noise_nodes))])))
        noise_samples = vals[noise_sample_idx]
        ## intervene in DAG
        self.scm.do({n: noise_samples[:, i] for i, n in enumerate(noise_nodes)})
        ## sample with these interventions
        counterfactual_samples = pd.DataFrame(self.scm.sample(n_samples), columns=self.scm.ordering)
        # construct cpts with new distribution
        self.counterfactual_model = self.model.copy()
        self._construct_CPD(counterfactual=True, df=counterfactual_samples)

    def exact_abduction_prediction(self, noi, ev, intn, n_joint_samples=30000):
        # sample from exact joint distribution
        start = time.time()
        joint = self.query(self.scm._get_exog_nodes(), ev)[0]
        values = np.array(list(product(*[range(card) for card in joint.cardinality])))
        n_joint_samples = max([n_joint_samples, 30*values.shape[0]])
        probabilities = joint.values.ravel()
        idx = np.random.choice(np.arange(values.shape[0]), size=n_joint_samples, p=probabilities)
        samples = values[idx]
        samples = {joint.variables[i]: samples[:, i] for i in range(len(joint.variables))}
        print(time.time() - start)
        # pass joint samples
        self.scm.do(samples)
        # format intervention
        if isinstance(intn[list(intn.keys())[0]], int):
            intn = {k: intn[k]*np.ones(n_joint_samples) for k in intn}
        self.scm.do(intn)
        # sample form new model
        prediction = self.scm.sample(return_pandas=True)[noi]
        return prediction.mean()

    def enumerate_inference(self, noi, ev, intn, n_samples=30000):
        """
        Performs exact counterfactual inference by enumeration.
        """
        intn = {k: intn[k]*np.ones(n_samples) for k in intn}
        joint_sample, joint_prob = self.posterior_enumerate(ev)
        joint_samples = joint_sample[np.random.choice(np.arange(joint_sample.shape[0]), p=joint_prob, size=n_samples)]
        joint_samples = {node: joint_samples[:, i] for i, node in enumerate(self.scm._get_exog_nodes())}
        self.scm.do(joint_samples)
        self.scm.do(intn)
        prediction = self.scm.sample(return_pandas=True)[noi]
        return prediction.mean()

    def posterior_enumerate(self, evidence):
        """
        Inference via enumeration.
        """
        # set up enumeration
        exog_nodes = self.scm._get_exog_nodes()
        endog_nodes = self.scm._get_endog_nodes()
        evidence_array = np.array([evidence[k] for k in endog_nodes if k in evidence])
        evidence_index = [i for i, v in enumerate(endog_nodes) if v in evidence]
        combinations = np.array(list(product(*[range(2) for _ in range(len(exog_nodes))])))
        probabilities = np.array([self.scm.G.nodes[node]['p'] for node in exog_nodes])
        prior = combinations * probabilities + (1 - combinations) * (1 - probabilities)

        def vector_compare(val_prob):
            joint_sample, prior = val_prob
            self.scm.do({exog_nodes[i]: joint_sample[i] for i in range(len(exog_nodes))})
            samp = self.scm.sample().flatten()
            if np.all(evidence_array == samp[evidence_index]):
                return np.product(prior)
            else:
                return 0

        posterior = np.array([i for i in map(vector_compare, zip(combinations, prior))])
        posterior = posterior / np.sum(posterior)
        return combinations, posterior

    def _generate_counterfactual_model(self, observed, intervention, n_samples=None):
        """
        Runs the standard counterfactual inference procedure and returns an intervened model with the posterior.

        Args:
            observed: a dictionary of {node: observed_value} to condition on.
            intervention: a dictionary of {node: intervention_value} to intervene on.
        """
        self.abduction(observed, n_samples)
        self.intervene(intervention, counterfactual=True)

    def standard_counterfactual_query(self, node_of_interest, observed, intervention, n_samples_for_approx=None):
        """
        Query and sample from the counterfactual model.
        Args:
            observed: a dictionary of {node: observed_value} to condition on.
            intervention: a dictionary of {node: intervention_value} to intervene on.
            n_samples: number of samples to draw from the counterfactual world model.
        """
        # infer latents and generate model, also initializes self.standard_inference_time
        self._generate_counterfactual_model(observed, intervention, n_samples=n_samples_for_approx)
        # then run the query
        ## for stability, pass in as evidence a deterministic value for the intervention node
        int_noise_node_values = {"U{}".format(k): intervention[k] for k in intervention}
        q, time_elapsed = self.query(node_of_interest, observed=int_noise_node_values, counterfactual=True)
        self.standard_inference_time = self.joint_inference_time + time_elapsed
        return q

    def merge_nodes(self, node_of_interest, intervention):
        """
        Merge nodes in the Twin Counterfactual network. In place modifies `self.twin_model`.
        Works by giving children of the node to be eliminated to its factual counterpart. Operates topologically.
        """
        # find every non-descendant of the intervention nodes
        nondescendant_sets = []
        all_nodes = set([i for i in list(self.model.nodes) if i[0] != 'U'])
        for node in intervention:
            nondescendant_sets.append(all_nodes.difference(set(nx.descendants(self.model, node))))
        dont_merge = [node_of_interest] + list(intervention.keys())
        shared_nondescendants = set.intersection(*nondescendant_sets) - set(dont_merge)
        # now modify twin network to replace all _tn variables with their regular counterpart
        ordered_nondescendants = [n for n in nx.topological_sort(self.model) if n in list(shared_nondescendants)]
        for node in ordered_nondescendants:  # start with the oldest nodes
            twin_node = node + "tn"
            tn_children = self.twin_model.successors(twin_node)
            self.twin_model.add_edges_from([(node, c) for c in tn_children])
            self.twin_model.remove_node(twin_node)

    def _eliminate_conditionally_independent(self, node_of_interest, observed, intervention):
        """
        Generate an "efficient" twin network model by removing nodes that are d-separated from the node
        of interest given observed and intervened variables.

        Args:
            node_of_interest: the node of interest in the query.
            observed: a dictionary of {node: observed_value} to condition on.
            intervention: a dictionary of {node: intervention_value} to intervene on.
        """
        conditioned_on = list(observed) + list(intervention)
        self.efficient_twin_model = self.twin_model.copy()
        for node in [n for n in self.twin_model.nodes if n[-2:] == "tn"]:
            try:
                if not self.efficient_twin_model.is_active_trail(node,
                                                                 node_of_interest + "tn",
                                                                 observed=conditioned_on):
                    self.efficient_twin_model.remove_node(node)
            except:
                pass

    def twin_counterfactual_query(self, node_of_interest, observed, intervention):
        """
        Query and sample from the counterfactual model.
        Args:
            observed: a dictionary of {node: observed_value} to condition on.
            intervention: a dictionary of {node: intervention_value} to intervene on.
            n_samples: number of samples to draw from the counterfactual world model.
        """
        self.create_twin_network(node_of_interest, observed, intervention)  # then, create the twin network
        result, time_elapsed = self.query(node_of_interest + "tn", observed, twin=True)  # log time it takes to do p(Vtn | E)
        return result

    def sample(self, n_samples=1, counterfactual=False, twin=False):
        """
        Perform forward sampling from the model.

        Args:
            n_samples: the number of samples you'd like to return
        """
        if counterfactual:
            model = self.counterfactual_model
        elif twin:
            model = self.twin_model
        else:
            model = self.model
        inference = BayesianModelSampling(model)
        return inference.forward_sample(size=n_samples, return_type='dataframe')

    def compare_times(self, node_of_interest, observed, intervention, n_samples_for_approx=None):
        """
        Compare the times it takes to do inference in the standard and twin network counterfactual inference
        approaches.

        Args:
            node_of_interest: the node of interest to perform inference on.
            observed: a dictionary of {node: observed_value} to condition on.
            intervention: a dictionary of {node: intervention_value} to intervene on.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("A. Performing Standard Counterfactual Inference.")
                self.standard_counterfactual_query(node_of_interest, observed, intervention, n_samples_for_approx)
                print("B. Performing Twin Network Counterfactual Inference.")
                # first, reset the graph network
                self.scm.G = self.scm.G_original.copy()
                self.twin_counterfactual_query(node_of_interest, observed, intervention)
                if self.verbose:
                    print(self.standard_inference_time, self.twin_inference_time)
                return self
        except Exception as e:
            print(e)
            print((node_of_interest, observed, intervention))
            return False  # return False bool to indicate failed experiment.