import warnings
import itertools
import torch
import math
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Torch tensor type default for consistency
torch.set_default_tensor_type(torch.DoubleTensor)

class CausalModel(object):
    """A class for building and handling Structural Causal Models.
    An SCM contains:
        - A graph G, made of a NetworkX DiGraph
        - Endogenous nodes X1, ..., Xn -- representing observable variables
        - Exogenous nodes U1, ..., Un -- one for each endogenous node
        - Functions f1, ..., fn -- that set Xi = fi(PAi, Ui) where PAi are the endogenous parents of Xi

    Has the ability to handle and operate on Twin Networks.

    Example:
        ```
        scm = CausalModel()
        scm.create(method="backward", n_nodes=5, max_parents=5)
        scm.create_twin_network()
        scm.do({"N3tn": 1})
        ```
    """
    def __init__(self, continuous=False, noise_coeff=1.):
        """Initialize a CausalModel instance.

        Args:
            continuous (bool): whether or not the graph is continuous or binary-valued. (Different inference schemes)
            noise_coeff (int/float): hyperparameter that scales the variance of noise.
        """
        self.twin_exists = False
        self.merged = False
        self.functions_assigned = False
        self.G = None
        self.G_original = None  # store an extra version of the original graph for resetting.
        self.continuous = continuous
        self.num_hidden = 20  # default number of hidden nodes per layer if using a neural network for functions.
        if noise_coeff < 0:
            self.noise_coeff = 1.
            warnings.warn("Noise coefficient was negative. Defaulting to 1.")
        else:
            self.noise_coeff = noise_coeff

    def create(self, method, n_nodes, **kwargs):
        """Creates a new random DAG according to the desired method.
        Use **kwargs to supply appropriate parameters to chosen `method`.

        Args:
            method: one of 'backward' (parameter: 'max_parents'), 'density' (parameter: 'prob'), or 'MCMC' (no params)
            n_nodes: number of nodes desired in the graph. 
        """
        assert method in ('backward', 'density', 'MCMC'), "method not in ('backward', 'density', 'MCMC')"
        if method == "backward":
            self.G = self.generate_DAG_backward(n_nodes, **kwargs)
        elif method == "density":
            self.G = self.generate_DAG_density(n_nodes, **kwargs)
        elif method == "MCMC":
            self.G = self.MCMC_generate_DAG(n_nodes, **kwargs)
        self._make_valid_graph()

    def generate_DAG_backward(self, n_nodes, max_parents):
        """
        Args:
            n_nodes: number of nodes desired in the graph.
            max_parents: the maximum number of parents per node in the graph.

        Returns:
            networkx.DiGraph: a networkx directed, acyclic graph (DAG).

        """
        nodes = ["N" + str(n) for n in range(n_nodes)]
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for i, node in enumerate(nodes[1:], start=1):
            nb_parents = np.random.randint(0, min([max_parents, i]) + 1)
            print(node, i, nb_parents)
            parent_nodes = [nodes[j] for j in np.random.choice(range(0, i), nb_parents, replace=False)]
            edges = [(pa, node) for pa in parent_nodes]
            graph.add_edges_from(edges)
        return graph

    def generate_DAG_density(self, n_nodes, prob):
        """Creates a graph from the RandomDAG algorithm given some neighbourhood connection probability.
        (Source: https://rdrr.io/cran/pcalg/man/randomDAG.html)

        Args:
            n_nodes (int): number of nodes in graph.
            prob (float): probability of downstream connections.
        """
        assert abs(prob) <= 1, "`prob` has to be in [0, 1]."
        nodes = ["N" + str(n) for n in range(n_nodes)]
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for i, node in enumerate(nodes):
            remaining_nodes = nodes[i + 1:]
            k = len(remaining_nodes)
            n_neighbours = np.random.binomial(k, prob)
            neighbours = np.random.choice(remaining_nodes,
                                          size=n_neighbours,
                                          replace=False)
            graph.add_edges_from([(node, n) for n in neighbours])
        return graph

    def MCMC_generate_DAG(self, n_nodes, **kwargs):
        """Creates a DAG by randomly adding or removing edges.
        
        Args:
            n_nodes: 
            **kwargs: 

        Returns:

        """
        nodes = ["N" + str(n) for n in range(n_nodes)]
        all_poss_edges = []

        for edge in itertools.permutations(nodes, 2):
            all_poss_edges.append(edge)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        loops = 5000

        for i in range(loops):
            idx_sample = np.random.randint(low=0, high=len(all_poss_edges))
            edge_chosen = all_poss_edges[idx_sample]

            if edge_chosen in list(graph.edges):
                graph.remove_edge(*edge_chosen)
            else:
                graph.add_edge(*edge_chosen)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(*edge_chosen)
                else:
                    pass
        return graph

    def create_from_DAG(self, G):
        """Create from an already existing DAG encoded as NetworkX graph.

        Args:
            G (nx graph): a networkx DAG.
        """
        assert nx.is_directed_acyclic_graph(G), "Graph is not a directed acyclic graph."
        self.G = G
        self._make_valid_graph()

    def _make_valid_graph(self):
        """Calls a sequence of functions that endows a NetworkX DAG with properties useful for a Structural
        Causal Model:
            1. Labels current nodes as endogenous
            2. Gives each endogenous variable an exogenous (noise) variable.
            3. Resets nodes to be unintervened with a null value.
            4. Gives all nodes their generating function.
            5. Store a copy of the graph for future reference.
            6. Labels the graph as a non-twin graph.
            7. Stores the original node order.
        """
        self._label_all_endogenous()
        self._add_noise_terms()
        self._label_all_nonintervened()
        self._label_all_valueless()
        self.ordering = sorted(self._get_exog_nodes(), key=lambda x: int(x[2:])) + self._get_endog_nodes()
        self._generate_all_functions()
        self.G_original = self.G.copy()
        self.G.is_twin = False


    def _label_all_endogenous(self):
        """Labels all nodes in the graph as endogenous."""
        for node in self.G.nodes:
            self.G.nodes[node]["endog"] = True

    def _label_all_nonintervened(self):
        """Labels all nodes in the graph as non-intervened."""
        for node in self.G.nodes:
            self.G.nodes[node]['intervened'] = False

    def _add_noise_terms(self):
        """Assign each endogenous variable an exogenous noise parent."""
        for node in list(self.G.nodes):
            noise_name = "U{}".format(str(node))
            self.G.add_node(noise_name)
            self.G.add_edge(noise_name, node)
            self.G.nodes[noise_name]["endog"] = False


    def _label_all_valueless(self):
        """Labels all nodes in the graph as without a value."""
        for node in self.G.nodes:
            self.G.nodes[node]['value'] = None

    def _create_twin_nodes(self):
        """Create the non-noise counterpart nodes in the Twin Network. This
        copies all attribute values -- i.e. endogeneity and functional form --
        over to the new network.
        """
        endog_nodes = [n for n in list(self.G.nodes.data())
                          if n[0] in self._get_endog_nodes()]
        endog_nodes = [(self._add_tn(n), d) for n, d in endog_nodes]
        self.twin_G.add_nodes_from(endog_nodes)

    def _create_twin_edges(self):
        """Creates the counterpart exogeneous and endogenous edges in the
        Twin Network.
        """
        shared_exog_edges = [(e[0], self._add_tn(e[1]))
                             for e in self._get_exog_edges()]

        # create the non-noise counterpart edges in the Twin network
        endog_edges = [(self._add_tn(e[0]), self._add_tn(e[1]))
                          for e in self._get_endog_edges()]

        self.twin_G.add_edges_from(endog_edges)
        self.twin_G.add_edges_from(shared_exog_edges)

    def create_twin_network(self):
        """Create a twin network by mirroring the original network, only sharing
        the noise terms.
        """
        assert self.functions_assigned, "Assign functions before creating TN."
        if not self.twin_exists:
            self.twin_G = self.G.copy()
            self._create_twin_nodes()
            self._create_twin_edges()
            self.twin_exists = True
            self.twin_G.is_twin = True

    def merge_in_twin(self, node_of_interest, intervention):
        """Merge nodes in the Twin Counterfactual network. In place creates & modifies `self.twin_G`.
        """
        # find every non-descendant of the intervention nodes
        if not self.merged:
            nondescendant_sets = []
            all_nodes = set([i for i in list(self.G.nodes) if i[0] != 'U'])
            for node in intervention:
                nondescendant_sets.append(all_nodes.difference(set(nx.descendants(self.G, node))))
            dont_merge = [node_of_interest] + list(intervention.keys())
            shared_nondescendants = set.intersection(*nondescendant_sets) - set(dont_merge)
            # now modify twin network to replace all _tn variables with their regular counterpart
            ordered_nondescendants = [n for n in nx.topological_sort(self.G) if n in list(shared_nondescendants)]
            for node in ordered_nondescendants:  # start with the oldest nodes
                twin_node = node + "tn"
                tn_children = self.twin_G.successors(twin_node)
                # TODO: This changes the ordering of nodes going into a node. This is currently fixed by a cheap hack
                # TODO:         which sorts edges alphabetically by when called. This should be fixed.
                self.twin_G.add_edges_from([(node, c) for c in tn_children])
                self.twin_G.remove_node(twin_node)
            print("Merging removed {} nodes.".format(len(ordered_nondescendants)))
            self.merged = True

    def draw(self, twin=False):
        """Draws/plots the network.

        Args:
            twin (bool): if true, plots the Twin Network.
        """
        G = self.twin_G if twin else self.G
        pos = nx.spring_layout(G)  # get layout positions for all nodes
        endog_nodes = self._get_endog_nodes(twin)
        exog_nodes = self._get_exog_nodes(twin)

        # draw nodes
        nx.draw_networkx_nodes(G,
                               pos,
                               nodelist=exog_nodes,
                               node_color='r',
                               node_size=500,
                               alpha=0.4,
                               label=exog_nodes)

        nx.draw_networkx_nodes(G,
                               pos,
                               nodelist=endog_nodes,
                               node_color='b',
                               node_size=800,
                               alpha=0.8)

        # draw edges
        nx.draw_networkx_edges(G,
                               pos,
                               width=1.0,
                               alpha=0.5,
                               arrowsize=20,
                               with_labels=True)

        # draw labels
        nx.draw_networkx_labels(G,
                                pos,
                                font_size=16,
                                font_color='white')

        plt.show()

    def _add_tn(self, node_name):
        """Helper function to modify the name of a node to its Twin Network
        version.

        Args:
            node_name (str): the name of the node.
        """
        return "{}tn".format(node_name)

    def _is_exog(self, node, g=None):
        """Checks if `node` is endogenous in graph `g`.
        Args:
            node (str): the name of the node
            g (nx.DiGraph): the graph
        """
        g = self.G if g is None else g
        return len(list(g.predecessors(node))) == 0 and not g.nodes[node]['endog']

    def _get_endog_nodes(self, twin=False):
        """Returns a list of the ids of endogenous nodes.
        
        Args:
            twin (bool): if True, use twin graph.
        """
        g = self.twin_G if self.twin_exists & twin else self.G
        return [node for node in nx.topological_sort(g) if not self._is_exog(node, g)]

    def _get_endog_edges(self):
        """Returns a list of the edges involving endogenous nodes."""
        return [e
                for e
                in filter(lambda x: self.G.nodes[x[0]]["endog"], self.G.edges)]

    def _get_exog_nodes(self, twin=False):
        """Returns a list of the ids of exogenous nodes.
        
        Args:
            twin (bool): if True, use twin graph.
        """
        g = self.twin_G if self.twin_exists & twin else self.G
        return [node for node in g.nodes if self._is_exog(node, g)]

    def _get_exog_edges(self):
        """Returns a list of edges involving exogenous nodes."""
        return [e
                for e
                in filter(lambda x: not self.G.nodes[x[0]]["endog"],
                          self.G.edges)]

    def _get_twin_nodes(self):
        """Returns an ordered set of twin nodes."""
        if not self.twin_exists:
            raise ValueError("Twin Network not yet created. Create one using .create_twin_network() first.")
        else:
            return [node for node in nx.topological_sort(self.twin_G) if "tn" in node]

    def _get_non_twin_endog_nodes(self):
        """Get endogenous nodes that are not twin nodes."""
        return [n for n in self._get_endog_nodes() if n[-2:] != "tn"]

    def _weave_sort_endog(self, evidence={}):
        """Weaves twin and non-twin nodes together ensuring primacy of observed nodes for inference.
        
        Args:
            evidence (dict): a dictionary of observed values formatted as {node_name: val}
        """
        twin_nodes = self._get_twin_nodes()
        non_twin = self._get_non_twin_endog_nodes()
        ordering = []
        for (nt, tw) in zip(non_twin, twin_nodes):
            if tw in evidence and nt not in evidence:
                ordering += [tw, nt]
            else:
                ordering += [nt, tw]
        return ordering

    def get_endog_order(self, evidence={}, twin=False):
        """Returns the order of endogenous nodes for probabilistic model creation in inference.
                
        Args:
            evidence (dict): a dictionary of observed values formatted as {node_name: val}
        """
        if twin:
            return self._weave_sort_endog(evidence)
        else:
            return self._get_endog_nodes()

    def _do_surgery(self, nodes):
        """
        Performs the do-operator graph surgery by removing all edges into each
        node in `nodes`.

        Args:
            nodes (nx node or list): a node, or a list of nodes, to perform
                                       the surgery on.
        """
        nodes = [nodes] if not isinstance(nodes, list) else nodes
        for node in nodes:
            parent_edges = [(pa, node) for pa in sorted(list(self.G.predecessors(node)))]
            self.G.remove_edges_from(parent_edges)

    def do(self, interventions):
        """
        Performs the intervention specified by the dictionary `interventions`.

        Args:
            interventions (dict): a dictionary of interventions of form {node_name: value}
        """
        nodes_to_intervene = list(interventions.keys())
        self._do_surgery(nodes_to_intervene)
        for node in nodes_to_intervene:
            self.G.nodes[node]['value'] = interventions[node]
            self.G.nodes[node]['intervened'] = True

    def _get_node_exog_parent(self, node):
        """
        Returns the name of the node's exogenous variable.
        
        Args:
            node (str): the name of the node.
        """
        parents = [p
                   for p
                   in sorted(list(self.G.predecessors(node)))
                   if self.G.nodes[p]['endog'] == False]
        assert len(parents) == 1, "More than one latent variable for node {node}. Should be an error; fix."
        return parents[0]


    def _generate_fn(self, node):
        """
        Generates a function for a given node.

        If not self.continuous:
            For endogenous nodes, the model is a logistic regression with parameters
            drawn from a standard normal. For exogenous nodes, the model is just
            the standard normal.
        else:
            Generates a 2 layer, tanh-activated neural network with `self.num_hidden` hidden layer nodes.

        Args:
            `node` (nx node): the index of the node to give a function
        """
        if self._is_exog(node, self.G):
            if self.continuous:
                self.G.nodes[node]['mu'] = np.random.normal(0, 0.3)
                self.G.nodes[node]['std'] = np.random.gamma(1, 0.4)
            else:
                self.G.nodes[node]['p'] = np.random.uniform(0.3, 0.7)
        else:
            n_parents = len([i for i in self.G.predecessors(node)])
            if self.continuous:
                if n_parents == 1:
                    self.G.nodes[node]['fn'] = lambda x: x
                else:
                    self.G.nodes[node]['fn'] = self._nn_function(n_parents)  # self._nn_function(n_parents
            else:
                theta = np.random.beta(5, 5, size=n_parents - 1)  # risk factors are positive & similar, thus beta.
                theta = theta / np.sum(theta)
                self.G.nodes[node]['parameters'] = theta

    def _nn_function(self, n_parents):
        """A helper function that generates a torch neural network model.
        
        Args:
            n_parents (int): the number of parents of a node.

        Returns:
            torch.nn.Sequential: the neural network.

        """
        layers = []
        num_hidden = self.num_hidden
        layers.append(torch.nn.modules.Linear(n_parents - 1, num_hidden))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.modules.Linear(num_hidden, 1))
        for l in layers:
            try:
                l.weight = torch.nn.Parameter(torch.randn(l.weight.shape, requires_grad=False))
            except:
                pass
            for p in l.parameters():
                p.requires_grad = False
        return torch.nn.Sequential(*layers)

    def _polynomial_function(self, n_parents, node):
        """Generates arbitrary polynomials of the form $\Sigma b_{n}x_{n}^c_{n}$
        
        Args:
            n_parents (int): the number of parents of a node.
            node (str): the node in question.
        """
        self.G.nodes[node]['betas'] = np.random.normal(size=n_parents - 1)
        self.G.nodes[node]['coefs'] = np.random.choice([1, 2, 4], size=n_parents-1, replace=True)
        return lambda x: (np.sum((self.G.nodes[node]['betas']*x) ** self.G.nodes[node]['coefs'], axis=1)\
                          - self.G.nodes[node]['mu_norm']) / self.G.nodes[node]['std_norm']

    def _calibrate_functions(self):
        """Calibrates the normalizers for use with the polynomial functional form."""
        for node in self._get_endog_nodes():
            samples = self.sample(200, return_pandas=True)
            self.G.nodes[node]['mu_norm'] = samples[node].mean()
            self.G.nodes[node]['std_norm'] = samples[node].std()

    def _generate_all_functions(self):
        """Gives all nodes a function."""
        for node in self.G.nodes:
            self.G.nodes[node]['mu_norm'] = np.array([0.])
            self.G.nodes[node]['std_norm'] = np.array([1.])
            self._generate_fn(node)
        self._calibrate_functions()
        self.functions_assigned = True

    def _sample_node(self, node, n_samples=1, graph=None):
        """Sample a value for the node.

        Args:
            node (str): the node to sample.
            n_samples (int): the number of samples to take.
            graph (nx.DiGraph): the graph to operate on (optional)
        """
        graph = self.G if graph is None else graph
        if graph.nodes[node]['intervened']:
            pass
        else:
            if self._is_exog(node, graph):  # if exogenous node
                if self.continuous:
                    mu = graph.nodes[node]['mu']
                    std = graph.nodes[node]['std']
                    graph.nodes[node]['value'] = self.noise_coeff * np.random.normal(mu, std, size=n_samples)
                else:
                    p = graph.nodes[node]['p']
                    graph.nodes[node]['value'] = np.random.binomial(1, p, size=n_samples)
            else:  # if an endogenous node
                parents = sorted(list(graph.predecessors(node)))
                endog_parents = [graph.nodes[n]['value'] for n in parents if n[0] != "U"]
                exog_parent = [graph.nodes[n]['value'] for n in parents if n[0] == "U"][0]
                if not endog_parents:
                    graph.nodes[node]['value'] = exog_parent   # take value from exogenous parent
                else:
                    fn = self._continuous_fn if self.continuous else self._binary_fn
                    val = fn(node, endog_parents, exog_parent, graph)
                    graph.nodes[node]['value'] = val
        return graph.nodes[node]['value']

    def _continuous_fn(self, node, parent_values, exog_value, graph=None):
        """A helper function that generates values for endogenous `node` given `parent_values` in the continuous case.

        Args:
            node (str): the node of interest.
            parent_values (list): the list of numpy arrays of parent values.
            graph (nx.DiGraph): the graph to operate on (optional)
        """
        if graph is None:
            graph = self.G
        X = np.hstack([pa.astype('float64').reshape(-1, 1) for pa in parent_values])
        X = torch.from_numpy(X)
        X_fn = graph.nodes[node]['fn']
        pred = X_fn(X).flatten().numpy()
        return pred + exog_value  # this is the additive noise function

    def _binary_fn(self, node, parent_values, exog_value, graph=None):
        """A helper function that generates values for endogenous `node` given `parent_values` in the binary case.

        Args:
            node (str): the node of interest.
            parent_values (list): the list of numpy arrays of parent values.
            graph (nx.DiGraph): the graph to operate on (optional)
        """
        if graph is None:
            graph = self.G
        thetas = graph.nodes[node]['parameters']
        v = np.dot(thetas, parent_values)
        if not isinstance(v, np.ndarray):
            v = 1. if v > 0.5 else 0.
        else:
            v = (v > 0.5).astype(np.float64)
        if not isinstance(v, np.ndarray):
            return v if not exog_value else 1. - v
        else:
            v[exog_value == 1] = 1 - v[exog_value == 1]
            return v

    def reset_graph(self):
        """Reset the graph to its original and destroy its twin companion."""
        self.G = self.G_original.copy()
        self.twin_G = None
        self.twin_exists = False

    def sample(self, n_samples=1, return_pandas=False, twin=False, evidence={}):
        """Sample from the full model.
        
        Args:
            n_samples (int): the number of samples to take.
            return_pandas (bool): whether to return a Pandas DF or not.
            twin (bool): whether to operate on the twin network graph.
            evidence (dict): a dictionary of observed values formatted as {node_name: val}

        Returns:
            samples: either a pandas dataframe or numpy array of samples.

        """
        graph = self.twin_G if (twin and self.twin_exists) else self.G
        if twin:
            ordering = self._get_exog_nodes() + self._weave_sort_endog(evidence)
        else:
            ordering = self.ordering
        for node in ordering:
            self._sample_node(node=node, n_samples=n_samples, graph=graph)
        samples = self._collect_samples(graph, ordering)
        if return_pandas:
            return pd.DataFrame(samples, columns=ordering)
        else:
            return samples

    def sample_observable_dict(self, n_samples, n_vars=None):
        """Return a dictionary of samples of (potentially partial) endogenous variables.
        Useful for generating arbitrary evidence sets for testing inference.

        Args:
            n_samples (int): the number of samples.
            n_vars (int): the number of randomly-chosen variables. If unset, it defaults to all endogenous vars.

        Returns:

        """
        d = self.sample(n_samples, return_pandas=True)
        d = d[self._get_exog_nodes()].to_dict("series")
        d = {k: torch.from_numpy(np.array(d[k], dtype=np.float64)) for k in d.keys()}
        if isinstance(n_vars, int):
            random_keys = np.random.choice(list(d.keys()), size=n_vars, replace=False)
            return {k: d[k] for k in d if k in random_keys}
        else:
            return d

    def _collect_samples(self, graph=None, ordering=None):
        """A helper function for ordering and stacking samples.

        Args:
            graph (nx.DiGraph): the graph to operate on.
            ordering:

        Returns:
            np.ndarray: an array of samples in the coorrect order.
        """
        ordering = self.ordering if ordering is None else ordering
        graph = self.G if graph is None else graph
        return np.vstack([graph.nodes[n]['value'] for n in ordering]).T

    def _sigmoid(self, x):
        """The sigmoid helper function.

        Args:
            x (numeric/array/etc.): input to sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def middle_node(self, node_of_interest):
        """Finds the node closest to the middle of the topological order that is an ancestor of `node_of_interest`.
        Useful for generating a node to intervene on for experimenting.

        Args:
            node_of_interest: the node of interest (i.e. the counterfactual outcome node)
        """
        endog_nodes = self._get_endog_nodes()
        limit = endog_nodes.index(node_of_interest)
        endog_nodes = endog_nodes[:limit]
        n = len(endog_nodes)
        ancestors = [a for a in nx.ancestors(self.G, node_of_interest) if a[0] != "U"]
        idx = np.argmin([abs(endog_nodes.index(a) - math.floor(n/2)) for a in ancestors])
        return ancestors[idx]
