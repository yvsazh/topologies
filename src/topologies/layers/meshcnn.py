import torch
from torch import nn


class MeshCNN(nn.Module):
    component_id = "meshcnn"

    def __init__(
        self,
        in_channels,
        out_channels,
        target_edges,
        normalization="batch",
        bias=True,
        num_groups=1,
    ):
        super().__init__()
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if not isinstance(target_edges, int) or isinstance(target_edges, bool) or target_edges <= 0:
            raise ValueError("target_edges must be a positive integer")
        if not isinstance(normalization, str) or normalization not in {"batch", "instance", "group", "none"}:
            raise ValueError("normalization must be batch, instance, group, or none")
        if bias is not True and bias is not False:
            raise ValueError("bias must be a boolean")
        if not isinstance(num_groups, int) or isinstance(num_groups, bool) or num_groups <= 0:
            raise ValueError("num_groups must be a positive integer")
        if normalization == "group" and out_channels % num_groups:
            raise ValueError("num_groups must divide out_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_edges = target_edges
        self.normalization = normalization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), bias=bias)
        if normalization == "batch":
            self.normalization_layer = nn.BatchNorm2d(out_channels)
        elif normalization == "instance":
            self.normalization_layer = nn.InstanceNorm2d(out_channels, affine=False)
        elif normalization == "group":
            self.normalization_layer = nn.GroupNorm(num_groups, out_channels)
        else:
            self.normalization_layer = nn.Identity()

    @staticmethod
    def _edge_key(first, second):
        return (first, second) if first < second else (second, first)

    def edge_convolution(self, edge_features, gemm_edges):
        if edge_features.ndim != 3 or edge_features.shape[1] != self.in_channels:
            raise ValueError("edge_features must have shape [batch, in_channels, edges]")
        if gemm_edges.ndim != 3 or gemm_edges.shape != (edge_features.shape[0], edge_features.shape[2], 4):
            raise ValueError("gemm_edges must have shape [batch, edges, 4]")
        batch_size, _, edge_count = edge_features.shape
        neighbors = gemm_edges.to(device=edge_features.device, dtype=torch.long)
        center_ids = torch.arange(edge_count, device=edge_features.device).view(1, edge_count, 1) + 1
        local_ids = torch.cat((center_ids.expand(batch_size, -1, -1), neighbors + 1), dim=2)
        offsets = torch.arange(batch_size, device=edge_features.device).view(batch_size, 1, 1) * (edge_count + 1)
        flat_ids = (local_ids + offsets).reshape(-1)
        features = edge_features.transpose(1, 2)
        padding = features.new_zeros((batch_size, 1, self.in_channels))
        flat_features = torch.cat((padding, features), dim=1).reshape(-1, self.in_channels)
        gathered = flat_features.index_select(0, flat_ids)
        gathered = gathered.view(batch_size, edge_count, 5, self.in_channels).permute(0, 3, 1, 2)
        center = gathered[:, :, :, 0]
        first_sum = gathered[:, :, :, 1] + gathered[:, :, :, 3]
        second_sum = gathered[:, :, :, 2] + gathered[:, :, :, 4]
        first_difference = torch.abs(gathered[:, :, :, 1] - gathered[:, :, :, 3])
        second_difference = torch.abs(gathered[:, :, :, 2] - gathered[:, :, :, 4])
        neighborhood = torch.stack(
            (center, first_sum, second_sum, first_difference, second_difference), dim=-1
        )
        return self.conv(neighborhood).squeeze(-1)

    def _validate_topology(self, edge_features, gemm_edges, sides, edges, edge_counts):
        tensors = (gemm_edges, sides, edges, edge_counts)
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("topology inputs and edge_counts must be tensors")
        if gemm_edges.ndim != 3 or gemm_edges.shape != (edge_features.shape[0], edge_features.shape[2], 4):
            raise ValueError("gemm_edges must have shape [batch, edges, 4]")
        if sides.shape != gemm_edges.shape:
            raise ValueError("sides must have the same shape as gemm_edges")
        if edges.shape != (edge_features.shape[0], edge_features.shape[2], 2):
            raise ValueError("edges must have shape [batch, edges, 2]")
        if edge_counts.shape != (edge_features.shape[0],):
            raise ValueError("edge_counts must have shape [batch]")
        for name, value in (("gemm_edges", gemm_edges), ("sides", sides), ("edges", edges), ("edge_counts", edge_counts)):
            if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError(f"{name} must use an integer dtype")
        gemm_values = gemm_edges.detach().cpu().tolist()
        side_values = sides.detach().cpu().tolist()
        edge_values = edges.detach().cpu().tolist()
        counts = edge_counts.detach().cpu().tolist()
        for batch_index, count in enumerate(counts):
            if not isinstance(count, int) or count <= 0 or count > edge_features.shape[2]:
                raise ValueError("each edge_count must be between 1 and the padded edge dimension")
            edge_rows = edge_values[batch_index][:count]
            gemm_rows = gemm_values[batch_index][:count]
            side_rows = side_values[batch_index][:count]
            keys = set()
            for edge_id, endpoints in enumerate(edge_rows):
                if len(endpoints) != 2 or endpoints[0] < 0 or endpoints[1] < 0 or endpoints[0] == endpoints[1]:
                    raise ValueError("edges must contain distinct nonnegative endpoint ids")
                key = self._edge_key(endpoints[0], endpoints[1])
                if key in keys:
                    raise ValueError("edges must not contain duplicate endpoint pairs")
                keys.add(key)
                for slot, neighbor in enumerate(gemm_rows[edge_id]):
                    reciprocal = side_rows[edge_id][slot]
                    if neighbor == -1:
                        if reciprocal != -1:
                            raise ValueError("boundary neighbors must use side index -1")
                    elif neighbor < 0 or neighbor >= count:
                        raise ValueError("gemm_edges contains an out-of-range edge index")
                    elif reciprocal < 0 or reciprocal > 3:
                        raise ValueError("sides contains an out-of-range reciprocal slot")
                    elif gemm_rows[neighbor][reciprocal] != edge_id or side_rows[neighbor][reciprocal] != slot:
                        raise ValueError("sides must identify reciprocal gemm_edges slots")
            faces = self._extract_faces(gemm_rows, edge_rows)
            if not faces:
                raise ValueError("topology must contain at least one triangular face")
            face_counts = {}
            for face in faces:
                for first, second in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])):
                    key = self._edge_key(first, second)
                    face_counts[key] = face_counts.get(key, 0) + 1
            if len(face_counts) != count or any(key not in face_counts or value > 2 for key, value in face_counts.items()):
                raise ValueError("active edges must belong to one or two triangular faces")
        return gemm_values, edge_values, counts

    def _extract_faces(self, gemm_rows, edge_rows):
        faces = set()
        for edge_id, neighbors in enumerate(gemm_rows):
            for first_slot, second_slot in ((0, 1), (2, 3)):
                first_neighbor = neighbors[first_slot]
                second_neighbor = neighbors[second_slot]
                if first_neighbor == -1 and second_neighbor == -1:
                    continue
                if first_neighbor < 0 or second_neighbor < 0:
                    raise ValueError("each triangular face must occupy a complete neighbor pair")
                if first_neighbor == second_neighbor or first_neighbor == edge_id or second_neighbor == edge_id:
                    raise ValueError("gemm_edges must identify three distinct edges per face")
                vertex_set = set(edge_rows[edge_id]) | set(edge_rows[first_neighbor]) | set(edge_rows[second_neighbor])
                if len(vertex_set) != 3:
                    raise ValueError("gemm_edges and edges do not describe triangular faces")
                face = tuple(sorted(vertex_set))
                faces.add(face)
        face_edges = {}
        for face in faces:
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])):
                key = self._edge_key(first, second)
                face_edges[key] = face_edges.get(key, 0) + 1
        if any(value > 2 for value in face_edges.values()):
            raise ValueError("topology is not a triangular manifold")
        return sorted(faces)

    def _can_collapse(self, edge_id, edge_rows, faces, groups):
        endpoint_map = {self._edge_key(*endpoints): key for key, endpoints in edge_rows.items()}
        if edge_id not in edge_rows:
            return None
        first, second = edge_rows[edge_id]
        incident_faces = [face for face in faces if first in face and second in face]
        if len(incident_faces) != 2:
            return None
        neighbors = {vertex: set() for pair in edge_rows.values() for vertex in pair}
        for left, right in edge_rows.values():
            neighbors[left].add(right)
            neighbors[right].add(left)
        shared = (neighbors[first] & neighbors[second]) - {first, second}
        opposite_vertices = {next(vertex for vertex in face if vertex not in {first, second}) for face in incident_faces}
        if shared != opposite_vertices or len(opposite_vertices) != 2:
            return None
        pairs = []
        ring_ids = {edge_id}
        for face in incident_faces:
            opposite = next(vertex for vertex in face if vertex not in {first, second})
            left_id = endpoint_map.get(self._edge_key(first, opposite))
            right_id = endpoint_map.get(self._edge_key(second, opposite))
            if left_id is None or right_id is None or left_id == right_id:
                return None
            pairs.append((left_id, right_id))
            ring_ids.update((left_id, right_id))
        face_edge_counts = {}
        for face in faces:
            for left, right in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])):
                key = endpoint_map.get(self._edge_key(left, right))
                if key is not None:
                    face_edge_counts[key] = face_edge_counts.get(key, 0) + 1
        if any(face_edge_counts.get(key, 0) != 2 for key in ring_ids):
            return None
        removed_ids = {edge_id}
        keep_ids = set()
        for left_id, right_id in pairs:
            keep_id, drop_id = sorted((left_id, right_id))
            keep_ids.add(keep_id)
            removed_ids.add(drop_id)
        remapped_faces = []
        for face in faces:
            remapped = tuple(sorted(first if vertex == second else vertex for vertex in face))
            if len(set(remapped)) == 3:
                remapped_faces.append(remapped)
        if len(set(remapped_faces)) != len(remapped_faces):
            return None
        remapped_edges = {}
        for key, (left, right) in edge_rows.items():
            if key in removed_ids:
                continue
            left = first if left == second else left
            right = first if right == second else right
            if left == right:
                return None
            new_key = self._edge_key(left, right)
            if new_key in remapped_edges:
                return None
            remapped_edges[new_key] = key
        if len(edge_rows) - len(removed_ids) != len(edge_rows) - 3:
            return None
        for keep_id in keep_ids:
            if keep_id not in groups:
                return None
        return first, second, pairs, removed_ids, remapped_faces

    def _build_topology(self, edge_rows, faces):
        ordered_ids = sorted(edge_rows)
        output_index = {edge_id: index for index, edge_id in enumerate(ordered_ids)}
        endpoint_map = {self._edge_key(*endpoints): edge_id for edge_id, endpoints in edge_rows.items()}
        slots = {edge_id: [] for edge_id in ordered_ids}
        for face in sorted(faces):
            first, second, third = face
            triangle_edges = (
                endpoint_map[self._edge_key(first, second)],
                endpoint_map[self._edge_key(second, third)],
                endpoint_map[self._edge_key(first, third)],
            )
            for position, edge_id in enumerate(triangle_edges):
                pair = (triangle_edges[(position + 1) % 3], triangle_edges[(position + 2) % 3])
                slots[edge_id].append(pair)
        gemm = [[-1, -1, -1, -1] for _ in ordered_ids]
        sides = [[-1, -1, -1, -1] for _ in ordered_ids]
        for edge_id in ordered_ids:
            if len(slots[edge_id]) > 2:
                raise ValueError("pooled topology is not a triangular manifold")
            flat_neighbors = [neighbor for pair in slots[edge_id] for neighbor in pair]
            if len(set(flat_neighbors)) != len(flat_neighbors):
                raise ValueError("pooled topology contains duplicate edge neighbors")
            output_edge = output_index[edge_id]
            for slot, neighbor_id in enumerate(flat_neighbors):
                gemm[output_edge][slot] = output_index[neighbor_id]
        for edge_id in ordered_ids:
            output_edge = output_index[edge_id]
            for slot, neighbor_index in enumerate(gemm[output_edge]):
                if neighbor_index < 0:
                    continue
                neighbor_id = ordered_ids[neighbor_index]
                neighbor_row = gemm[output_index[neighbor_id]]
                reciprocal_slots = [index for index, value in enumerate(neighbor_row) if value == output_edge]
                if len(reciprocal_slots) != 1:
                    raise ValueError("pooled topology has invalid reciprocal edge slots")
                sides[output_edge][slot] = reciprocal_slots[0]
        packed_edges = [list(self._edge_key(*edge_rows[edge_id])) for edge_id in ordered_ids]
        return packed_edges, gemm, sides

    def _pool_one(self, features, gemm_rows, input_edges, count):
        edge_rows = {edge_id: tuple(input_edges[edge_id]) for edge_id in range(count)}
        faces = self._extract_faces(gemm_rows[:count], input_edges[:count])
        groups = {edge_id: {edge_id} for edge_id in range(count)}
        priority = torch.sum(features.detach() * features.detach(), dim=0).cpu().tolist()
        candidates = sorted(range(count), key=lambda edge_id: (priority[edge_id], edge_id))
        if (count - self.target_edges) % 3:
            raise ValueError("target_edges must match a sequence of three-edge triangular collapses")
        current_count = count
        for edge_id in candidates:
            if current_count <= self.target_edges:
                break
            collapse = self._can_collapse(edge_id, edge_rows, faces, groups)
            if collapse is None:
                continue
            first, second, pairs, removed_ids, new_faces = collapse
            if current_count - 3 < self.target_edges:
                continue
            for left_id, right_id in pairs:
                keep_id, _ = sorted((left_id, right_id))
                groups[keep_id] = groups[edge_id] | groups[left_id] | groups[right_id]
                edge_rows[keep_id] = (first, next(vertex for vertex in (edge_rows[left_id] + edge_rows[right_id]) if vertex not in {first, second}))
            for removed_id in removed_ids:
                edge_rows.pop(removed_id, None)
                groups.pop(removed_id, None)
            for active_id, endpoints in list(edge_rows.items()):
                left, right = endpoints
                left = first if left == second else left
                right = first if right == second else right
                edge_rows[active_id] = self._edge_key(left, right)
            faces = new_faces
            current_count -= 3
        if current_count != self.target_edges:
            raise ValueError("valid edge collapses cannot reach target_edges for this topology")
        packed_edges, pooled_gemm, pooled_sides = self._build_topology(edge_rows, faces)
        output_features = torch.stack(
            [features[:, sorted(groups[edge_id])].mean(dim=1) for edge_id in sorted(edge_rows)], dim=1
        )
        return output_features, pooled_gemm, pooled_sides, packed_edges

    def _normalize_active(self, convolution, counts):
        if self.normalization == "batch":
            active = torch.cat(
                [convolution[index, :, :count] for index, count in enumerate(counts)], dim=1
            )
            normalized = self.normalization_layer(active.unsqueeze(0).unsqueeze(-1))
            normalized = normalized.squeeze(0).squeeze(-1)
            return [
                torch.relu(value)
                for value in torch.split(normalized, counts, dim=1)
            ]
        activated = []
        for batch_index, count in enumerate(counts):
            active = convolution[batch_index : batch_index + 1, :, :count].unsqueeze(-1)
            normalized = self.normalization_layer(active).squeeze(-1).squeeze(0)
            activated.append(torch.relu(normalized))
        return activated

    def forward(self, edge_features, gemm_edges, sides, edges, edge_counts=None, return_topology=False):
        if not isinstance(edge_features, torch.Tensor) or edge_features.ndim != 3:
            raise ValueError("edge_features must have shape [batch, in_channels, edges]")
        if edge_features.shape[1] != self.in_channels:
            raise ValueError("edge_features channel count must match in_channels")
        if edge_features.shape[0] <= 0 or edge_features.shape[2] <= 0:
            raise ValueError("edge_features must contain at least one mesh and one edge")
        if not edge_features.is_floating_point():
            raise ValueError("edge_features must use a floating point dtype")
        if return_topology is not True and return_topology is not False:
            raise ValueError("return_topology must be a boolean")
        if edge_counts is None:
            edge_counts = torch.full(
                (edge_features.shape[0],), edge_features.shape[2], device=edge_features.device, dtype=torch.long
            )
        elif isinstance(edge_counts, int) and not isinstance(edge_counts, bool):
            edge_counts = torch.full(
                (edge_features.shape[0],), edge_counts, device=edge_features.device, dtype=torch.long
            )
        elif isinstance(edge_counts, torch.Tensor):
            if edge_counts.ndim == 0 and edge_features.shape[0] == 1:
                edge_counts = edge_counts.reshape(1)
            elif edge_counts.shape == (edge_features.shape[0], 1):
                edge_counts = edge_counts.reshape(-1)
        gemm_values, edge_values, counts = self._validate_topology(
            edge_features, gemm_edges, sides, edges, edge_counts
        )
        if any(self.target_edges > count for count in counts):
            raise ValueError("target_edges cannot exceed any active edge_count")
        convolution = self.edge_convolution(edge_features, gemm_edges.to(device=edge_features.device))
        activated = self._normalize_active(convolution, counts)
        batch_features = []
        batch_gemm = []
        batch_sides = []
        batch_edges = []
        for batch_index, count in enumerate(counts):
            pooled, pooled_gemm, pooled_sides, pooled_edges = self._pool_one(
                activated[batch_index],
                gemm_values[batch_index],
                edge_values[batch_index],
                count,
            )
            batch_features.append(pooled)
            batch_gemm.append(pooled_gemm)
            batch_sides.append(pooled_sides)
            batch_edges.append(pooled_edges)
        pooled_features = torch.stack(batch_features, dim=0)
        topology_device = edge_features.device
        pooled_gemm_edges = torch.tensor(batch_gemm, dtype=torch.long, device=topology_device)
        pooled_sides = torch.tensor(batch_sides, dtype=torch.long, device=topology_device)
        pooled_edges = torch.tensor(batch_edges, dtype=torch.long, device=topology_device)
        pooled_counts = torch.full(
            (edge_features.shape[0],), self.target_edges, dtype=torch.long, device=topology_device
        )
        self.pooled_topology = (pooled_gemm_edges, pooled_sides, pooled_edges, pooled_counts)
        if return_topology:
            return (pooled_features, *self.pooled_topology)
        return pooled_features
