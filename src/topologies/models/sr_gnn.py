import torch
from torch import nn


def _build_session_graph(sessions, padding_idx, dtype):
    batch_size, time_steps = sessions.shape
    item_ids = torch.full_like(sessions, padding_idx)
    valid = torch.zeros_like(sessions, dtype=torch.bool)
    incoming = torch.zeros(batch_size, time_steps, time_steps, dtype=dtype, device=sessions.device)
    outgoing = torch.zeros_like(incoming)
    last_nodes = torch.zeros(batch_size, dtype=torch.long, device=sessions.device)

    for batch_index, row in enumerate(sessions.detach().cpu().tolist()):
        sequence = [item for item in row if item != padding_idx]
        ordered_items = list(dict.fromkeys(sequence))
        node_for_item = {item: node for node, item in enumerate(ordered_items)}
        node_count = len(ordered_items)
        item_ids[batch_index, :node_count] = torch.tensor(
            ordered_items, dtype=torch.long, device=sessions.device
        )
        valid[batch_index, :node_count] = True
        last_nodes[batch_index] = node_for_item[sequence[-1]]

        for source_item, target_item in zip(sequence, sequence[1:]):
            source = node_for_item[source_item]
            target = node_for_item[target_item]
            outgoing[batch_index, source, target] += 1
            incoming[batch_index, target, source] += 1
        incoming[batch_index] /= incoming[batch_index].sum(dim=-1, keepdim=True).clamp_min(1)
        outgoing[batch_index] /= outgoing[batch_index].sum(dim=-1, keepdim=True).clamp_min(1)

    return item_ids, valid, incoming, outgoing, last_nodes


class SRGNN(nn.Module):
    architecture_id = "sr-gnn"

    def __init__(self, num_items=32, embedding_dim=16, propagation_steps=1, padding_idx=0):
        super().__init__()
        if num_items < 2 or embedding_dim < 1 or propagation_steps < 1:
            raise ValueError("num_items, embedding_dim, and propagation_steps must be positive")
        if not 0 <= padding_idx < num_items:
            raise ValueError("padding_idx must be in [0, num_items)")
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.propagation_steps = propagation_steps
        self.padding_idx = padding_idx
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx)
        self.incoming = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.outgoing = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gate = nn.GRUCell(2 * embedding_dim, embedding_dim)
        self.attention_item = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_target = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_score = nn.Linear(embedding_dim, 1, bias=False)
        self.session_projection = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    def forward(self, sessions):
        sessions = torch.as_tensor(sessions, dtype=torch.long, device=self.item_embedding.weight.device)
        if sessions.ndim != 2 or sessions.shape[1] < 1:
            raise ValueError("sessions must have shape (batch, time) with time >= 1")
        if torch.any(sessions < 0) or torch.any(sessions >= self.num_items):
            raise ValueError("item IDs must be in [0, num_items)")
        if torch.any(sessions.ne(self.padding_idx).sum(dim=1) == 0):
            raise ValueError("each session must contain at least one non-padding item")

        item_ids, valid, incoming, outgoing, last_nodes = _build_session_graph(
            sessions, self.padding_idx, self.item_embedding.weight.dtype
        )
        nodes = self.item_embedding(item_ids)
        for _ in range(self.propagation_steps):
            incoming_messages = self.incoming(torch.bmm(incoming, nodes))
            outgoing_messages = self.outgoing(torch.bmm(outgoing, nodes))
            message = torch.cat((incoming_messages, outgoing_messages), dim=-1)
            nodes = self.gate(
                message.reshape(-1, 2 * self.embedding_dim),
                nodes.reshape(-1, self.embedding_dim),
            ).reshape_as(nodes)
            nodes = nodes * valid.unsqueeze(-1)

        target = nodes[torch.arange(sessions.shape[0], device=sessions.device), last_nodes]
        attention = self.attention_score(
            torch.sigmoid(self.attention_item(nodes) + self.attention_target(target).unsqueeze(1))
        ).squeeze(-1)
        attention = attention.masked_fill(~valid, torch.finfo(attention.dtype).min)
        weights = torch.softmax(attention, dim=1)
        global_preference = torch.sum(nodes * weights.unsqueeze(-1), dim=1)
        session_vector = self.session_projection(torch.cat((global_preference, target), dim=-1))
        return session_vector @ self.item_embedding.weight.transpose(0, 1)
