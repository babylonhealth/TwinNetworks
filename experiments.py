import pickle
import numpy as np
import networkx as nx
from scm import CausalModel
from exact import ExactCounterfactual
from concurrent.futures import ProcessPoolExecutor, wait


class Experiment(object):
    """
    A helper class to generate and run many different types of experiments for the twin networks paper.
    (And for counterfactual inference in general.)
    """
    def __init__(self, merge=False):
        self.merge = merge

    def _get_ordering(self, cfi):
        return list(nx.topological_sort(cfi.model))

    def _get_endog_nodes(self, cfi):
        return [i for i in self._get_ordering(cfi) if i[0] != "U"]

    def _choose_node_of_interest(self, cfi):
        ordering = self._get_endog_nodes(cfi)
        return np.random.choice(ordering[2:])

    def _choose_evidence_nodes(self, cfi, p=0.3):
        endog_nodes = self._get_endog_nodes(cfi)
        k = max([np.random.binomial(len(endog_nodes), p), 1])
        return np.random.choice(endog_nodes, k, replace=False)

    def _generate_evidence(self, nodes, p=0.5):
        return {k: np.random.binomial(1, p) for k in nodes}


    def _choose_intervention_nodes_complex(self, node_of_interest):
        """
        Choose nodes to intervene upon. Note this is slightly more complex than just randomly choosing nodes, for
        two reasons:
            a) You cannot intervene on the node of interest
            b) No intervention can d-separate another intervention and the node of interest. This prevents the
               "Node X is not in the graph." error.

        Args:
            cfi: the Counterfactual class instance.
            node_of_interest: the node of interest.
        """
        int_nodes = [i for i in self._get_endog_nodes(cfi) if i != node_of_interest]
        int_number = min([max([np.random.poisson(1), 1]), len(int_nodes)])
        shuffled = np.random.permutation(int_nodes)
        chosen_ints = 0
        idx = 0
        interventions = {}
        while chosen_ints < int_number or idx < len(int_nodes):  # while you have fewer than desired ints, and still candidates
            candidate_node = shuffled[idx] # create a candidate
            # check if candidate d-separates any other interventions from NOI
            # if not, add to graph
            idx += 1
        raise NotImplementedError
        # return interventions

    def _choose_intervention_nodes(self, cfi, node_of_interest):
        """
        Chooses a single intervention node that has an active path to the node of interest in the intervened,
        unconditioned model.

        Args:
            cfi: the Counterfactual class instance.
            node_of_interest: the node of interest.
        """
        def check_interventional_active_trail(model, intervention_node, node_of_interest):
            """
            Checks if there is an active path between the intervention node and the node of interest in the
            intervened model.
            """
            synthetic_model = model.copy()
            int_edges = [e for e in synthetic_model.edges if e[1] == intervention_node]
            synthetic_model.remove_edges_from(int_edges)
            return synthetic_model.is_active_trail(intervention_node, node_of_interest)

        int_nodes = [n for n in self._get_endog_nodes(cfi) if n != node_of_interest
                     and check_interventional_active_trail(cfi.model, n, node_of_interest)]

        if len(int_nodes) == 0:
            # note: if this ever happens, it will throw an error down the line, as intervening on some random node
            # will d-separate the node from the node of interest. This is because of the pgmpy elimination order
            # algorithm, which should be fixed. Maybe just raise an error for now.
            raise ValueError("Could not find a node to intervene on with {} as a descendant.".format(node_of_interest))
            # return np.random.choice([i for i in self._get_endog_nodes(cfi) if i != node_of_interest], 1).tolist()

        return np.random.choice(int_nodes, 1).tolist()  # otherwise, choose a random node to intervene on.

    def _generate_intervention(self, nodes, observed):
        proposed_intervention = self._generate_evidence(nodes)
        for node in proposed_intervention:
            if node in observed:
                proposed_intervention[node] = 1 - observed[node]
        return proposed_intervention

    def get_times(self, cfi):
        return cfi.joint_inference_time, cfi.standard_inference_time, cfi.twin_inference_time

    def _generate_graph_size(self, size):
        """
        Generates a graph of a specific size.

        Args:
            size: the number of endogenous nodes in the graph.
        """
        scm = CausalModel()
        scm.create("MCMC", n_nodes=size)
        return scm

    def _generate_graph_density(self, size, density=0.5):
        """
        Generates a graph of a specific size and density.

        Args:
            size: the number of endogeneous nodes in the graph.
            density: the graph density, i.e. p(edge(i->j)) for all i > j in topological order.
        """
        scm = CausalModel()
        multi_parents = False
        while not multi_parents:  # ensures that there are more than two parents, to ensure it's not d-separated by intn
            scm.create("density", n_nodes=size, prob=density)
            noi = [n for n in list(nx.topological_sort(scm.G)) if n[0] != "U"][-1]
            multi_parents = len(list(scm.G.predecessors(noi))) > 1
        return scm

    def _generate_counterfactual_instance(self, scm):
        """
        Generates a causal model and constructs CPDs from sampling from `scm`.

        Args:
            scm: A CausalModel instance.
        """
        cfi = ExactCounterfactual()
        cfi.construct(scm)
        return cfi

    def size_experiment(self, size):
        """
        Experiment to test the effect of graph size on inference time.
        Args:
            size:
        """
        scm = self._generate_graph_density(size)
        cfi = self._generate_counterfactual_instance(scm)
        node_of_interest = self._choose_node_of_interest(cfi)
        evidence_nodes = self._choose_evidence_nodes(cfi)
        evidence = self._generate_evidence(evidence_nodes)
        intervention_nodes = self._choose_intervention_nodes(cfi, node_of_interest)
        intervention = self._generate_intervention(intervention_nodes, evidence)
        return scm, node_of_interest, evidence, intervention

    def density_experiment(self, density, graph_size=10):
        """
        Experiment to test the effect of density on inference time.

        Args:
            density: value in [0, 1] of desired graph density.
            size: the size of the graph.
        """
        scm = self._generate_graph_density(graph_size, density=density)
        cfi = self._generate_counterfactual_instance(scm)
        node_of_interest = self._choose_node_of_interest(cfi)
        evidence_nodes = self._choose_evidence_nodes(cfi)
        evidence = self._generate_evidence(evidence_nodes)
        intervention_nodes = self._choose_intervention_nodes(cfi, node_of_interest)
        intervention = self._generate_intervention(intervention_nodes, evidence)
        return scm, node_of_interest, evidence, intervention

    def late_evidence_size_experiment(self, p, graph_size=10):
        """
        Experiment to evaluate the effect of the number of nodes to set evidence on.

        Using parameter p, it chooses the k ~ Bin(|G|, p) nodes from the bottom of the topographic order
        to set evidence for.

        Args:
            p: size parameter.
            graph_size: size of the graph to evaluate with.
        """
        scm = self._generate_graph_size(graph_size)
        cfi = self._generate_counterfactual_instance(scm)
        node_of_interest = self._choose_node_of_interest(cfi)
        node_ordering = self._get_endog_nodes(cfi)[::-1]
        k = max([np.random.binomial(len(node_ordering), p), 1])
        evidence_nodes = node_ordering[:k]
        evidence = self._generate_evidence(evidence_nodes)
        intervention_nodes = self._choose_intervention_nodes(cfi, node_of_interest)
        intervention = self._generate_intervention(intervention_nodes, evidence)
        return scm, node_of_interest, evidence, intervention

    def evidence_topography_experiment(self, p, graph_size=10):
        """
        Experiment to test the effect of the location of evidence on inference time.
        It creates an evidence set equal to half of the nodes in the graph, and shifts that
        continually from the top to the bottom of the topological order.
        Args:
            p: parameter determining location of evidence. p=0 -> top of order, p=1 -> bottom.
            graph_size: size of the graph.
        """
        scm = self._generate_graph_size(graph_size)
        cfi = self._generate_counterfactual_instance(scm)
        node_of_interest = self._choose_node_of_interest(cfi)
        node_ordering = self._get_endog_nodes(cfi)
        evidence_start = np.random.binomial(len(node_ordering), p)
        evidence_size = min([len(node_ordering) - evidence_start,
                             int(len(node_ordering) / 2.)])
        evidence_nodes = node_ordering[evidence_start:evidence_size]
        evidence = self._generate_evidence(evidence_nodes)
        intervention_nodes = self._choose_intervention_nodes(cfi, node_of_interest)
        intervention = self._generate_intervention(intervention_nodes, evidence)
        return scm, node_of_interest, evidence, intervention

    def intervention_topography_experiment(self, p, graph_size=10):
        """
        Experiment to test the effect of the location of the intervention on inference.

        Args:
            p:
            graph_size:
        """
        # randomly choose int / ev prior to edge construction
        scm = self._generate_graph_size(graph_size)
        cfi = self._generate_counterfactual_instance(scm)
        node_of_interest = self._choose_node_of_interest(cfi)  # assumption: choose last
        node_ordering = self._get_endog_nodes(cfi)
        k = max([np.random.binomial(len(node_ordering), 0.25), 1])
        evidence_nodes = node_ordering[::-1][:k]  # assumption: choose low
        evidence = self._generate_evidence(evidence_nodes)
        k_intervention = np.random.binomial(len(node_ordering), p)
        intervention_node = [node_ordering[k_intervention]]
        intervention = self._generate_intervention(intervention_node, evidence)
        return scm, node_of_interest, evidence, intervention

    def compare_times_parallel(self, scm, noi, ev, intn):
        """
        Implemented as a middle-man function to take care of creating a counterfactual instance in a new worker.
        Created because I hypothesize the passing of a CFI is leading to pickle errors, leading to catastrophic
        failure when running.

        Args:
            scm: A CausalModel instance.
            noi: the node of interest.
            ev: a dictionary of evidence.
            intn: a dictionary of interventions.

        Returns:

        """
        cfi = ExactCounterfactual(merge=self.merge)
        cfi.construct(scm)
        return cfi.compare_times(noi, ev, intn)

    def run_experiment(self, experiment_fn, params, num_trials, file_name, workers, **kwargs):
        """
        A big nasty function. Make sexier!

        The master experiment runner/controller. Takes an experiment function, generates the conditions per trial,
        runs the trials in parallel across parameter values, and stores/saves.

        Args:
            params: an array of parameter values to vary.
            num_trials: the number of successful trials per parameter value.
            experiment_fn: the experiment condition generating function.
            file_name: the filename to save to.
            **kwargs: keywords to pass to `experiment_fn`.
        """
        print("Now running: {}".format(file_name))
        experiment_dict = {}
        for p in params:
            print("\n\n\n\n\nNow trying experiment {}".format(p))
            experiment_dict[p] = {}
            experiment_dict[p]["joint"] = []
            experiment_dict[p]["standard"] = []
            experiment_dict[p]['prediction'] = []
            experiment_dict[p]["twin"] = []
            num_successful = 0
            previous_num_successful = 0
            num_failures = 0
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = []
                    num_running = 0
                    while num_successful < num_trials:
                        if len(futures) != 0:
                            finished = [f for f in filter(lambda x: x.done(), futures)]  # finished trials
                            results = [r.result() for r in finished] if len(finished) > 0 else []
                            num_successful = len([r for r in results if not isinstance(r, bool)])  # num successful
                            num_running = len([f for f in filter(lambda x: not x.done(), futures)])  # running trials
                        if num_successful + num_running < num_trials and num_running < workers:  # if successful trials + candidates < desired, start
                            try:  # if graph construction fails
                                scm, node_of_interest, evidence, intervention = experiment_fn(p, **kwargs)
                                val_process = executor.submit(self.compare_times_parallel, scm, node_of_interest, evidence, intervention)
                                futures.append(val_process)
                            except Exception as e:
                                print(e)
                                num_failures += 1
                        if num_successful > previous_num_successful:
                            previous_num_successful = num_successful
                            print("S: {}\t| R: {}".format(num_successful, num_running))
                wait(futures)
                times = [self.get_times(f.result()) for f in futures if not isinstance(f.result(), bool)]
            else:
                times = []
                for i in range(num_trials):
                    try:
                        scm, node_of_interest, evidence, intervention = experiment_fn(p, **kwargs)
                        cfi = self.compare_times_parallel(scm, node_of_interest, evidence, intervention)
                        times.append((cfi.joint_inference_time, cfi.standard_inference_time, cfi.twin_inference_time))
                    except Exception as e:
                        print(e)
                        num_failures += 1
            for j, s, t in times:
                experiment_dict[p]['joint'].append(j)
                experiment_dict[p]['standard'].append(s)
                experiment_dict[p]['prediction'].append(s - j)
                experiment_dict[p]['twin'].append(t)
            # experiment_dict[p]['failures'] = len(futures) - num_successful + num_failures  # add failed returns to failed constructions
        with open("../../results/{}.pkl".format(file_name), "wb") as f:
            pickle.dump(experiment_dict, f)
        return experiment_dict


class InterventionExperiments(Experiment):
    def __init__(self, merge=True):
        Experiment.__init__(self, merge=merge)

    def _choose_node_of_interest(self, cfi, p=0.):
        nodes = self._get_endog_nodes(cfi)
        k = np.random.binomial(len(nodes), p=p)
        return nodes[min([k, len(nodes) - 1])]

    def _choose_evidence_nodes(self, cfi, p=0.):
        nodes = self._get_endog_nodes(cfi)
        k = np.random.binomial(len(nodes), p=p)
        if k - 1 + 3 < len(nodes):
            return nodes[-3:]
        else:
            return nodes[k:k+3]

    def intervention_topography_experiment(self, params, graph_size=10):
        """
        Experiment to test the effect of the location of the intervention on inference.

        Args:
            p:
            graph_size:
        """
        noi_loc, ev_loc, int_loc = params
        # randomly choose int / ev prior to edge construction?
        scm = self._generate_graph_size(graph_size)
        cfi = self._generate_counterfactual_instance(scm)
        node_of_interest = self._choose_node_of_interest(cfi, noi_loc)
        node_ordering = self._get_endog_nodes(cfi)
        evidence_nodes = self._choose_evidence_nodes(cfi, ev_loc)
        evidence = self._generate_evidence(evidence_nodes)
        k_intervention = np.random.binomial(len(node_ordering), int_loc)
        intervention_node = [node_ordering[min([k_intervention, len(node_ordering) - 1])]]
        intervention = self._generate_intervention(intervention_node, evidence)
        return scm, node_of_interest, evidence, intervention