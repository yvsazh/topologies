import torch
from torch import nn


class OTCFM(nn.Module):
    architecture_id = "ot-cfm"
    def __init__(self, features=4, hidden_features=64):
        super().__init__()
        self.features = features
        self.network = nn.Sequential(
            nn.Linear(features + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, features),
        )

    @staticmethod
    def couple(data, noise):
        if data.ndim < 2 or noise.ndim != data.ndim or data.shape != noise.shape:
            raise ValueError("data and noise must have equal batched shapes")
        batch = data.shape[0]
        if batch == 0:
            raise ValueError("data and noise batches must be non-empty")
        flat_data = data.reshape(batch, -1)
        flat_noise = noise.reshape(batch, -1)
        costs = torch.cdist(flat_data, flat_noise).square()
        assignment = OTCFM._minimum_cost_assignment(costs)
        return noise[assignment], assignment

    @staticmethod
    def _minimum_cost_assignment(costs):
        values = costs.detach().cpu().tolist()
        size = len(values)
        row_potential = [0.0] * (size + 1)
        column_potential = [0.0] * (size + 1)
        matched_row = [0] * (size + 1)
        predecessor = [0] * (size + 1)
        for row in range(1, size + 1):
            matched_row[0] = row
            column = 0
            minimum = [float("inf")] * (size + 1)
            visited = [False] * (size + 1)
            while True:
                visited[column] = True
                current_row = matched_row[column]
                delta = float("inf")
                next_column = 0
                for candidate in range(1, size + 1):
                    if visited[candidate]:
                        continue
                    reduced = (values[current_row - 1][candidate - 1]
                               - row_potential[current_row] - column_potential[candidate])
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        predecessor[candidate] = column
                    if minimum[candidate] < delta:
                        delta = minimum[candidate]
                        next_column = candidate
                for candidate in range(size + 1):
                    if visited[candidate]:
                        row_potential[matched_row[candidate]] += delta
                        column_potential[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matched_row[column] == 0:
                    break
            while True:
                previous = predecessor[column]
                matched_row[column] = matched_row[previous]
                column = previous
                if column == 0:
                    break
        assignment = [0] * size
        for column in range(1, size + 1):
            assignment[matched_row[column] - 1] = column - 1
        return torch.tensor(assignment, dtype=torch.long, device=costs.device)

    def forward(self, state, time):
        if state.shape[-1] != self.features:
            raise ValueError("state final dimension must equal features")
        time = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        if time.ndim == 0:
            time = time.expand(*state.shape[:-1], 1)
        elif time.shape == state.shape[:-1]:
            time = time.unsqueeze(-1)
        else:
            time = torch.broadcast_to(time, (*state.shape[:-1], 1))
        return self.network(torch.cat((state, time), dim=-1))

    @staticmethod
    def interpolate(data, paired_noise, time):
        return (1 - time) * paired_noise + time * data

    @staticmethod
    def target_velocity(data, paired_noise):
        return data - paired_noise

    def training_pair(self, data, noise, time):
        paired_noise, indices = self.couple(data, noise)
        state = self.interpolate(data, paired_noise, time)
        return state, self.target_velocity(data, paired_noise), indices
