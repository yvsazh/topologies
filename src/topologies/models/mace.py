import itertools
import math

import torch
from torch import nn


def _wigner_3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    triangle = math.sqrt(
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )
    magnetic = math.sqrt(
        math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )
    lower = max(0, j2 - j3 - m1, j1 - j3 + m2)
    upper = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = 0.0
    for z in range(lower, upper + 1):
        denominator = (
            math.factorial(z)
            * math.factorial(j1 + j2 - j3 - z)
            * math.factorial(j1 - m1 - z)
            * math.factorial(j2 + m2 - z)
            * math.factorial(j3 - j2 + m1 + z)
            * math.factorial(j3 - j1 - m2 + z)
        )
        total += (-1.0) ** z / denominator
    return (-1.0) ** (j1 - j2 - m3) * triangle * magnetic * total


def _clebsch_gordan(j1, j2, j3):
    values = []
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            values.append([
                (-1.0) ** (j1 - j2 + m3)
                * math.sqrt(2 * j3 + 1)
                * _wigner_3j(j1, j2, j3, m1, m2, -m3)
                for m3 in range(-j3, j3 + 1)
            ])
    return torch.tensor(values, dtype=torch.float64).reshape(2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1)


def _spherical_harmonics(unit_vectors, max_ell):
    x = unit_vectors[..., 2]
    phi = torch.atan2(unit_vectors[..., 1], unit_vectors[..., 0])
    harmonics = []
    for ell in range(max_ell + 1):
        components = []
        for m in range(ell + 1):
            pmm = torch.ones_like(x)
            if m:
                odd_double_factorial = 1
                for value in range(1, 2 * m, 2):
                    odd_double_factorial *= value
                pmm = ((-1.0) ** m) * odd_double_factorial * torch.clamp(1.0 - x.square(), min=0.0).pow(m / 2)
            if ell == m:
                associated = pmm
            else:
                previous = pmm
                current = (2 * m + 1) * x * pmm
                if ell == m + 1:
                    associated = current
                else:
                    for degree in range(m + 2, ell + 1):
                        following = ((2 * degree - 1) * x * current - (degree + m - 1) * previous) / (degree - m)
                        previous, current = current, following
                    associated = current
            normalization = math.sqrt(
                (2 * ell + 1)
                * math.factorial(ell - m)
                / math.factorial(ell + m)
            )
            amplitude = associated * normalization
            phase = m * phi
            positive = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
            components.append(positive)
            if m:
                components.insert(0, ((-1.0) ** m) * positive.conj())
        harmonics.append(torch.stack(components, dim=-1))
    return harmonics


def _correlation_paths(max_ell, correlation):
    paths = []
    for order in range(1, correlation + 1):
        for ells in itertools.product(range(max_ell + 1), repeat=order):
            states = [(ells[0], ())]
            for ell in ells[1:]:
                states = [
                    (target, intermediates + (target,))
                    for previous, intermediates in states
                    for target in range(abs(previous - ell), previous + ell + 1)
                ]
            paths.extend(
                (ells, intermediates, target)
                for target, intermediates in states
                if target <= max_ell
            )
    return paths


def _value_error(message):
    return ValueError(message)


class MACE(nn.Module):
    architecture_id = "mace"
    PRESETS = {
        "tiny": {
            "num_interactions": 2,
            "hidden_channels": 4,
            "max_ell": 1,
            "correlation": 2,
            "num_radial_basis": 4,
            "cutoff": 5.0,
            "cutoff_p": 5,
            "radial_hidden": [8],
            "readout_hidden": 8,
            "energy_scale": 1.0,
            "energy_shift": 0.0,
        },
        "small": {
            "num_interactions": 2,
            "hidden_channels": 16,
            "max_ell": 2,
            "correlation": 3,
            "num_radial_basis": 8,
            "cutoff": 5.0,
            "cutoff_p": 5,
            "radial_hidden": [32, 32],
            "readout_hidden": 16,
            "energy_scale": 1.0,
            "energy_shift": 0.0,
        },
        "base": {
            "num_interactions": 2,
            "hidden_channels": 256,
            "max_ell": 3,
            "correlation": 3,
            "num_radial_basis": 8,
            "cutoff": 5.0,
            "cutoff_p": 5,
            "radial_hidden": [64, 64, 64, 1024],
            "readout_hidden": 16,
            "energy_scale": 1.0,
            "energy_shift": 0.0,
        },
    }

    def __init__(
        self,
        num_species,
        num_interactions=2,
        hidden_channels=8,
        max_ell=1,
        correlation=2,
        num_radial_basis=8,
        cutoff=5.0,
        cutoff_p=5,
        radial_hidden=(16, 16),
        readout_hidden=16,
        energy_scale=1.0,
        energy_shift=0.0,
    ):
        super().__init__()
        integer_values = {
            "num_species": num_species,
            "num_interactions": num_interactions,
            "hidden_channels": hidden_channels,
            "max_ell": max_ell,
            "correlation": correlation,
            "num_radial_basis": num_radial_basis,
            "cutoff_p": cutoff_p,
            "readout_hidden": readout_hidden,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise _value_error(f"{name} must be an integer")
            if value <= 0 and name != "max_ell":
                raise ValueError(f"{name} must be positive")
        if max_ell < 0 or max_ell > 3:
            raise ValueError("max_ell must be between 0 and 3")
        if correlation > 3:
            raise ValueError("correlation must be between 1 and 3")
        if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be a finite positive number")
        for name, value in (("energy_scale", energy_scale), ("energy_shift", energy_shift)):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if isinstance(radial_hidden, (str, bytes)):
            raise _value_error("radial_hidden must be a nonempty sequence of positive integers")
        try:
            radial_hidden = tuple(radial_hidden)
        except TypeError as exc:
            raise ValueError("radial_hidden must be a nonempty sequence of positive integers") from exc
        if not radial_hidden or any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in radial_hidden):
            raise _value_error("radial_hidden must be a nonempty sequence of positive integers")

        self.num_species = num_species
        self.num_interactions = num_interactions
        self.hidden_channels = hidden_channels
        self.max_ell = max_ell
        self.correlation = correlation
        self.num_radial_basis = num_radial_basis
        self.cutoff = float(cutoff)
        self.cutoff_p = cutoff_p
        self.paths = _correlation_paths(max_ell, correlation)
        self.embedding = nn.Embedding(num_species, hidden_channels)
        self.first_radial = nn.Linear(num_radial_basis, hidden_channels, bias=False)
        self.radial_networks = nn.ModuleList()
        self.radial_projections = nn.ModuleList()
        self.channel_mixing = nn.ModuleList()
        self.product_mixing = nn.ModuleList()
        self.path_weights = nn.ParameterList()
        self.residual_weights = nn.ParameterList()
        self.interaction_readouts = nn.ModuleList()
        triple_paths = [
            (ell1, ell2, target)
            for ell1 in range(max_ell + 1)
            for ell2 in range(max_ell + 1)
            for target in range(abs(ell1 - ell2), min(ell1 + ell2, max_ell) + 1)
        ]
        self.triple_paths = triple_paths
        for interaction in range(num_interactions):
            widths = [num_radial_basis, *radial_hidden]
            layers = []
            for input_width, output_width in itertools.pairwise(widths):
                layers.append(nn.Linear(input_width, output_width))
                layers.append(nn.SiLU())
            self.radial_networks.append(nn.Sequential(*layers))
            self.radial_projections.append(nn.ModuleList([
                nn.Linear(radial_hidden[-1], hidden_channels) for _ in triple_paths
            ]))
            self.channel_mixing.append(nn.ParameterList([
                nn.Parameter(torch.eye(hidden_channels)) for _ in range(max_ell + 1)
            ]))
            self.product_mixing.append(nn.ParameterList([
                nn.Parameter(torch.eye(hidden_channels))
                for _ in range(correlation * (max_ell + 1))
            ]))
            self.path_weights.append(nn.Parameter(torch.empty(num_species, len(self.paths), hidden_channels)))
            nn.init.normal_(self.path_weights[-1], std=hidden_channels ** -0.5)
            self.residual_weights.append(nn.Parameter(torch.eye(hidden_channels).repeat(num_species, max_ell + 1, 1, 1)))
            self.interaction_readouts.append(nn.Linear(hidden_channels, 1))

        self.initial_readout = nn.Linear(hidden_channels, 1)
        self.final_readout = nn.Sequential(
            nn.Linear(hidden_channels, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, 1),
        )
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        coupling_keys = set()
        for ell1, ell2, target in triple_paths:
            coupling_keys.add((ell1, ell2, target))
        for ells, intermediates, _ in self.paths:
            if len(ells) > 1:
                previous = ells[0]
                for index, ell in enumerate(ells[1:]):
                    target = intermediates[index]
                    coupling_keys.add((previous, ell, target))
                    previous = target
        self._coupling_names = {}
        for left, right, target in sorted(coupling_keys):
            name = f"cg_{left}_{right}_{target}"
            self.register_buffer(name, _clebsch_gordan(left, right, target), persistent=False)
            self._coupling_names[(left, right, target)] = name

    def radial_basis(self, distances):
        if not isinstance(distances, torch.Tensor) or not distances.is_floating_point():
            raise ValueError("distances must be a floating-point tensor")
        if not torch.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("distances must be finite and nonnegative")
        index = torch.arange(1, self.num_radial_basis + 1, device=distances.device, dtype=distances.dtype)
        scaled = distances.unsqueeze(-1) * index * (1.0 / self.cutoff)
        basis = math.sqrt(2.0 / self.cutoff) * (index * math.pi / self.cutoff) * torch.sinc(scaled)
        ratio = distances / self.cutoff
        p = self.cutoff_p
        envelope = (
            1.0
            - ((p + 1) * (p + 2) / 2.0) * ratio.pow(p)
            + p * (p + 2) * ratio.pow(p + 1)
            - (p * (p + 1) / 2.0) * ratio.pow(p + 2)
        )
        return basis * envelope.unsqueeze(-1) * (distances < self.cutoff).unsqueeze(-1)

    @staticmethod
    def correlation_products(values, order):
        if not isinstance(values, torch.Tensor) or values.ndim < 1:
            raise ValueError("values must be a tensor with a neighbor dimension")
        if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
            raise ValueError("order must be a positive integer")
        return values.sum(dim=0).pow(order)

    @staticmethod
    def energy_force_loss(
        predicted_energy,
        target_energy,
        predicted_force,
        target_force,
        lambda_energy=1.0,
        lambda_force=1000.0,
    ):
        if predicted_energy.shape != target_energy.shape:
            raise ValueError("predicted_energy and target_energy must have matching shapes")
        if predicted_force.shape != target_force.shape or predicted_force.shape[-1] != 3:
            raise ValueError("predicted_force and target_force must have matching [..., 3] shapes")
        return lambda_energy * (predicted_energy - target_energy).square().mean() + lambda_force * (
            predicted_force - target_force
        ).square().mean()

    def _validate_inputs(self, positions, atom_types, batch):
        if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [num_atoms, 3]")
        if not positions.is_floating_point() or positions.shape[0] == 0:
            raise ValueError("positions must be a nonempty floating-point tensor")
        if not torch.isfinite(positions).all():
            raise ValueError("positions must contain only finite values")
        if not isinstance(atom_types, torch.Tensor) or atom_types.ndim != 1 or atom_types.shape[0] != positions.shape[0]:
            raise ValueError("atom_types must have shape [num_atoms]")
        if atom_types.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("atom_types must use an integer dtype")
        if atom_types.device != positions.device:
            raise ValueError("positions and atom_types must be on the same device")
        if (atom_types < 0).any() or (atom_types >= self.num_species).any():
            raise ValueError("atom_types must be between 0 and num_species - 1")
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
        if not isinstance(batch, torch.Tensor) or batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
            raise ValueError("batch must have shape [num_atoms]")
        if batch.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("batch must use an integer dtype")
        if batch.device != positions.device:
            raise ValueError("positions and batch must be on the same device")
        if (batch < 0).any():
            raise ValueError("batch graph IDs must be nonnegative")
        return batch.to(dtype=torch.long)

    def _edges(self, positions, batch):
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = torch.linalg.vector_norm(relative, dim=-1)
        count = positions.shape[0]
        mask = (batch[:, None] == batch[None, :]) & (distances < self.cutoff)
        mask = mask & ~torch.eye(count, dtype=torch.bool, device=positions.device)
        target, source = mask.nonzero(as_tuple=True)
        return target, source, relative[target, source], distances[target, source]

    def _initial_features(self, positions, atom_types, batch):
        target, source, relative, distances = self._edges(positions, batch)
        count = positions.shape[0]
        complex_dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64
        features = [
            torch.zeros(count, self.hidden_channels, 2 * ell + 1, dtype=complex_dtype, device=positions.device)
            for ell in range(self.max_ell + 1)
        ]
        if target.numel() == 0:
            return features
        radial = self.first_radial(self.radial_basis(distances))
        edge_channels = radial * self.embedding(atom_types[source])
        safe_distance = distances.clamp_min(torch.finfo(distances.dtype).eps)
        harmonics = _spherical_harmonics(relative / safe_distance.unsqueeze(-1), self.max_ell)
        for ell, harmonic in enumerate(harmonics):
            edge_features = edge_channels.to(complex_dtype).unsqueeze(-1) * harmonic.unsqueeze(1)
            features[ell] = features[ell].index_add(0, target, edge_features)
        return features

    def _couple(self, left, right, left_ell, right_ell, target_ell):
        name = self._coupling_names[(left_ell, right_ell, target_ell)]
        coefficients = getattr(self, name).to(dtype=left.dtype)
        return torch.einsum("bcm,bcn,mnp->bcp", left, right, coefficients)

    def _interaction(self, features, positions, atom_types, batch, interaction):
        target, source, relative, distances = self._edges(positions, batch)
        count = positions.shape[0]
        complex_dtype = features[0].dtype
        aggregate = [
            torch.zeros(count, self.hidden_channels, 2 * ell + 1, dtype=complex_dtype, device=positions.device)
            for ell in range(self.max_ell + 1)
        ]
        if target.numel() == 0:
            return self._correlate(aggregate, atom_types, interaction)
        radial = self.radial_networks[interaction](self.radial_basis(distances))
        safe_distance = distances.clamp_min(torch.finfo(distances.dtype).eps)
        harmonics = _spherical_harmonics(relative / safe_distance.unsqueeze(-1), self.max_ell)
        for path_index, (ell1, ell2, target_ell) in enumerate(self.triple_paths):
            mixed = torch.einsum(
                "bcm,kc->bkm",
                features[ell2][source],
                self.channel_mixing[interaction][ell2].to(complex_dtype),
            )
            radial_weight = self.radial_projections[interaction][path_index](radial)
            angular = harmonics[ell1].to(complex_dtype)
            edge_features = mixed * radial_weight.to(complex_dtype).unsqueeze(-1)
            edge_features = edge_features.unsqueeze(-2) * angular.unsqueeze(1).unsqueeze(-1)
            coefficients = getattr(self, self._coupling_names[(ell1, ell2, target_ell)]).to(dtype=complex_dtype)
            messages = torch.einsum("bcmn,mnp->bcp", edge_features, coefficients)
            aggregate[target_ell] = aggregate[target_ell].index_add(0, target, messages)
        return self._correlate(aggregate, atom_types, interaction)

    def _correlate(self, features, atom_types, interaction):
        count = atom_types.shape[0]
        complex_dtype = features[0].dtype
        outputs = [
            torch.zeros(count, self.hidden_channels, 2 * ell + 1, dtype=complex_dtype, device=atom_types.device)
            for ell in range(self.max_ell + 1)
        ]
        weighted = {}
        for factor in range(self.correlation):
            for ell in range(self.max_ell + 1):
                weight = self.product_mixing[interaction][factor * (self.max_ell + 1) + ell].to(complex_dtype)
                weighted[(factor, ell)] = torch.einsum("bcm,kc->bkm", features[ell], weight)
        for path_index, (ells, intermediates, target) in enumerate(self.paths):
            value = weighted[(0, ells[0])]
            previous = ells[0]
            for factor, ell in enumerate(ells[1:], start=1):
                next_ell = intermediates[factor - 1]
                value = self._couple(value, weighted[(factor, ell)], previous, ell, next_ell)
                previous = next_ell
            species_weights = self.path_weights[interaction][atom_types, path_index]
            outputs[target] = outputs[target] + value * species_weights.to(complex_dtype).unsqueeze(-1)
        updated = []
        for ell, message in enumerate(outputs):
            residual = torch.einsum(
                "bcm,bkc->bkm",
                features[ell],
                self.residual_weights[interaction][atom_types, ell].to(complex_dtype),
            )
            updated.append(message + residual)
        return updated

    def site_scale_shift(self, site_energies):
        return self.energy_scale * site_energies + self.energy_shift

    def forward(self, positions, atom_types, batch=None):
        batch = self._validate_inputs(positions, atom_types, batch)
        features = self._initial_features(positions, atom_types, batch)
        site_energies = self.initial_readout(features[0].real.squeeze(-1)).squeeze(-1)
        for interaction in range(self.num_interactions):
            features = self._interaction(features, positions, atom_types, batch, interaction)
            readout = self.interaction_readouts[interaction](features[0].real.squeeze(-1)).squeeze(-1)
            if interaction == self.num_interactions - 1:
                readout = readout + self.final_readout(features[0].real.squeeze(-1)).squeeze(-1)
            site_energies = site_energies + readout
        site_energies = self.site_scale_shift(site_energies)
        graph_count = int(batch.max().item()) + 1
        return positions.new_zeros(graph_count).index_add(0, batch, site_energies)
