"""Runnable local structured-data examples (no network or optional packages)."""
import torch
from topologies.models.matrix_factorization import MatrixFactorization
from topologies.models.graph_network import GraphNetwork
from topologies.models.actor_critic import ActorCritic
from topologies.models.q_network import QNetwork
from topologies.training.recipes.value_learning import TDValueRecipe


def matrix_factorization_holdout():
    # Train-only positive/negative pairs; test negative pool is disjoint.
    torch.manual_seed(11)
    model = MatrixFactorization(5, 8, 8, score_mode="ranking")
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    user, pos, neg = torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])
    for _ in range(25):
        optimizer.zero_grad(); loss = torch.nn.functional.softplus(model(user, neg) - model(user, pos)).mean(); loss.backward(); optimizer.step()
    heldout = model(torch.tensor([0, 1, 2, 3]), torch.tensor([5, 6, 7, 4]))
    return loss.detach(), heldout.detach()


def karate_style_graph():
    # Small undirected toy topology encoded as directed source->target arcs.
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    x = torch.eye(4); y = torch.tensor([0, 0, 1, 1]); train_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    validation_mask = ~train_mask
    model = GraphNetwork(4, 8, 2)
    opt = torch.optim.Adam(model.parameters(), lr=.05)
    for _ in range(20):
        opt.zero_grad(); logits = model(x, edges); loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask]); loss.backward(); opt.step()
    return logits[validation_mask].detach()


def discrete_bandit_step():
    model = ActorCritic(2, 2, (8,)); obs = torch.tensor([[1., 0.], [0., 1.]])
    logits, values = model(obs); actions = torch.tensor([0, 1]); rewards = torch.tensor([1., -1.])
    loss = -torch.distributions.Categorical(logits=logits).log_prob(actions).mul(rewards).mean() + values.square().mean()
    loss.backward()
    return loss.detach()


def td_transition_example():
    rewards = torch.tensor([1., 1.]); next_q = torch.tensor([[5., 2.], [5., 2.]])
    # First transition terminates; second merely truncates and continues bootstrapping.
    return TDValueRecipe.td_target(rewards, next_q, torch.tensor([True, False]), .9, torch.tensor([False, True]))


if __name__ == "__main__":
    print("MF", matrix_factorization_holdout())
    print("graph validation logits", karate_style_graph())
    print("bandit loss", discrete_bandit_step())
    print("TD target", td_transition_example())
    print("Q values", QNetwork(2, 2)(torch.zeros(1, 2)))
