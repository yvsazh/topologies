import math
from numbers import Real

import torch
from torch import nn


class OCSORT(nn.Module):
    component_id = "oc-sort"

    @staticmethod
    def _is_real_in_range(value, minimum, maximum=None):
        if not isinstance(value, Real) or isinstance(value, bool):
            return False
        number = float(value)
        return (
            math.isfinite(number)
            and number >= minimum
            and (maximum is None or number <= maximum)
        )

    @staticmethod
    def _is_boolean(value):
        return isinstance(value, bool)

    def __init__(
        self,
        detection_threshold=0.6,
        iou_threshold=0.3,
        delta_t=3,
        inertia=0.2,
        max_age=30,
        min_hits=3,
        use_byte=False,
    ):
        super().__init__()
        if not self._is_real_in_range(detection_threshold, 0.0, 1.0):
            raise ValueError("detection_threshold must be a finite number in [0, 1]")
        if not self._is_real_in_range(iou_threshold, 0.0, 1.0):
            raise ValueError("iou_threshold must be a finite number in [0, 1]")
        if not isinstance(delta_t, int) or isinstance(delta_t, bool) or delta_t < 1:
            raise ValueError("delta_t must be a positive integer")
        if not self._is_real_in_range(inertia, 0.0):
            raise ValueError("inertia must be a finite non-negative number")
        if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age < 0:
            raise ValueError("max_age must be a non-negative integer")
        if not isinstance(min_hits, int) or isinstance(min_hits, bool) or min_hits < 1:
            raise ValueError("min_hits must be a positive integer")
        if not self._is_boolean(use_byte):
            raise ValueError("use_byte must be a boolean")
        if use_byte:
            raise ValueError("use_byte=True is outside the OC-SORT core implementation")
        self.detection_threshold = float(detection_threshold)
        self.iou_threshold = float(iou_threshold)
        self.delta_t = delta_t
        self.inertia = float(inertia)
        self.max_age = max_age
        self.min_hits = min_hits

    @staticmethod
    def _box_to_measurement(box):
        if not isinstance(box, torch.Tensor) or box.shape != (4,):
            raise ValueError("box must be a tensor with shape [4]")
        if not torch.isfinite(box).all().item() or (box[2:] <= box[:2]).any().item():
            raise ValueError("box must have finite coordinates and positive width and height")
        width = box[2] - box[0]
        height = box[3] - box[1]
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        return torch.stack((center_x, center_y, width * height, width / (height + 1e-6)))

    @staticmethod
    def _state_to_box(state):
        area = state[2].clamp_min(1e-12)
        ratio = state[3].clamp_min(1e-12)
        width = torch.sqrt(area * ratio)
        height = area / width
        return torch.stack(
            (
                state[0] - width * 0.5,
                state[1] - height * 0.5,
                state[0] + width * 0.5,
                state[1] + height * 0.5,
            )
        )

    @staticmethod
    def _motion_matrix(reference):
        matrix = torch.eye(7, dtype=reference.dtype, device=reference.device)
        matrix[0, 4] = 1.0
        matrix[1, 5] = 1.0
        matrix[2, 6] = 1.0
        return matrix

    @staticmethod
    def _measurement_matrix(reference):
        matrix = reference.new_zeros((4, 7))
        matrix[:, :4] = torch.eye(4, dtype=reference.dtype, device=reference.device)
        return matrix

    @staticmethod
    def _process_noise(reference):
        return torch.diag(reference.new_tensor([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01]))

    @staticmethod
    def _measurement_noise(reference):
        return torch.diag(reference.new_tensor([1.0, 1.0, 10.0, 10.0]))

    @classmethod
    def _kalman_predict(cls, state, covariance):
        motion = cls._motion_matrix(state)
        predicted_state = motion @ state
        predicted_covariance = motion @ covariance @ motion.mT + cls._process_noise(state)
        return predicted_state, predicted_covariance

    @staticmethod
    def _kalman_update(state, covariance, measurement, observation_matrix, measurement_noise):
        innovation_covariance = observation_matrix @ covariance @ observation_matrix.mT + measurement_noise
        gain = torch.linalg.solve(innovation_covariance, observation_matrix @ covariance).mT
        innovation = measurement - observation_matrix @ state
        updated_state = state + gain @ innovation
        identity = torch.eye(state.shape[0], dtype=state.dtype, device=state.device)
        residual = identity - gain @ observation_matrix
        updated_covariance = residual @ covariance @ residual.mT + gain @ measurement_noise @ gain.mT
        return updated_state, (updated_covariance + updated_covariance.mT) * 0.5

    @staticmethod
    def _box_iou(first, second):
        top_left = torch.maximum(first[:2], second[:2])
        bottom_right = torch.minimum(first[2:], second[2:])
        intersection = (bottom_right - top_left).clamp_min(0.0).prod()
        first_area = (first[2:] - first[:2]).clamp_min(0.0).prod()
        second_area = (second[2:] - second[:2]).clamp_min(0.0).prod()
        union = first_area + second_area - intersection
        return intersection / union.clamp_min(1e-12)

    def _ocm_similarity(self, prior_box, latest_box, predicted_box, detection_box, confidence):
        iou = self._box_iou(predicted_box, detection_box)
        prior_center = (prior_box[:2] + prior_box[2:]) * 0.5
        latest_center = (latest_box[:2] + latest_box[2:]) * 0.5
        detection_center = (detection_box[:2] + detection_box[2:]) * 0.5
        track_direction = (latest_center - prior_center)[[1, 0]]
        detection_direction = (detection_center - prior_center)[[1, 0]]
        track_norm = torch.linalg.vector_norm(track_direction)
        detection_norm = torch.linalg.vector_norm(detection_direction)
        if track_norm.item() == 0.0 or detection_norm.item() == 0.0:
            consistency = iou.new_zeros(())
        else:
            cosine = torch.dot(track_direction, detection_direction) / (track_norm * detection_norm)
            angle = torch.acos(cosine.clamp(-1.0, 1.0))
            consistency = (math.pi * 0.5 - angle.abs()) / math.pi
        confidence_value = torch.as_tensor(confidence, dtype=iou.dtype, device=iou.device)
        bonus = self.inertia * confidence_value * consistency
        return iou, consistency, bonus, iou + bonus

    @staticmethod
    def _interpolate_boxes(first_box, last_box, num_intermediate):
        if not isinstance(num_intermediate, int) or isinstance(num_intermediate, bool) or num_intermediate < 0:
            raise ValueError("num_intermediate must be a non-negative integer")
        if num_intermediate == 0:
            return []
        return [
            first_box + (last_box - first_box) * (step / (num_intermediate + 1))
            for step in range(1, num_intermediate + 1)
        ]

    def _oru_reupdate(self, state, covariance, first_box, last_box, num_intermediate):
        virtual_boxes = self._interpolate_boxes(first_box, last_box, num_intermediate)
        observation_matrix = self._measurement_matrix(state)
        measurement_noise = self._measurement_noise(state)
        for box in (*virtual_boxes, last_box):
            state, covariance = self._kalman_predict(state, covariance)
            state, covariance = self._kalman_update(
                state,
                covariance,
                self._box_to_measurement(box),
                observation_matrix,
                measurement_noise,
            )
        return state, covariance

    @staticmethod
    def _hungarian(cost):
        size = len(cost)
        if size == 0:
            return []
        row_potential = [0.0] * (size + 1)
        column_potential = [0.0] * (size + 1)
        matched_row = [0] * (size + 1)
        previous_column = [0] * (size + 1)
        for row in range(1, size + 1):
            matched_row[0] = row
            minimum = [float("inf")] * (size + 1)
            used = [False] * (size + 1)
            column = 0
            while True:
                used[column] = True
                current_row = matched_row[column]
                delta = float("inf")
                next_column = 0
                for candidate in range(1, size + 1):
                    if used[candidate]:
                        continue
                    reduced = cost[current_row - 1][candidate - 1] - row_potential[current_row] - column_potential[candidate]
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        previous_column[candidate] = column
                    if minimum[candidate] < delta:
                        delta = minimum[candidate]
                        next_column = candidate
                for candidate in range(size + 1):
                    if used[candidate]:
                        row_potential[matched_row[candidate]] += delta
                        column_potential[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matched_row[column] == 0:
                    break
            while True:
                previous = previous_column[column]
                matched_row[column] = matched_row[previous]
                column = previous
                if column == 0:
                    break
        assignment = [-1] * size
        for column in range(1, size + 1):
            if matched_row[column] != 0:
                assignment[matched_row[column] - 1] = column - 1
        return assignment

    def _associate(self, frame, detection_indices, tracks, use_ocm):
        num_detections = len(detection_indices)
        num_tracks = len(tracks)
        if num_detections == 0 or num_tracks == 0:
            return []
        internal_boxes = [
            frame[index, :4].detach().to(dtype=torch.float64)
            for index in detection_indices
        ]
        affinities = [[0.0] * num_tracks for _ in range(num_detections)]
        ious = [[0.0] * num_tracks for _ in range(num_detections)]
        for detection_position, detection_box in enumerate(internal_boxes):
            for track_position, track in enumerate(tracks):
                if use_ocm:
                    prior_box = self._historical_box(track)
                    iou, _, _, affinity = self._ocm_similarity(
                        prior_box,
                        track["last_box"],
                        track["predicted_box"],
                        detection_box,
                        frame[detection_indices[detection_position], 4].detach().to(dtype=torch.float64),
                    )
                else:
                    iou = self._box_iou(track["last_box"], detection_box)
                    affinity = iou
                ious[detection_position][track_position] = float(iou.item())
                affinities[detection_position][track_position] = float(affinity.item())
        size = num_detections + num_tracks
        costs = [[0.0] * size for _ in range(size)]
        match_bias = self.inertia + 1.0 if use_ocm else 1.0
        for detection_position in range(num_detections):
            for track_position in range(num_tracks):
                costs[detection_position][track_position] = -(
                    affinities[detection_position][track_position] + match_bias
                )
        assignment = self._hungarian(costs)
        matches = []
        for detection_position in range(num_detections):
            track_position = assignment[detection_position]
            if 0 <= track_position < num_tracks and ious[detection_position][track_position] >= self.iou_threshold:
                matches.append((detection_indices[detection_position], track_position))
        return matches

    def _historical_box(self, track):
        if len(track["history"]) <= 1:
            return track["last_box"]
        target_frame = track["last_frame"] - self.delta_t
        historical = [item for item in track["history"][:-1] if item[0] <= target_frame]
        if historical:
            return historical[-1][1]
        return track["history"][0][1]

    @staticmethod
    def _new_track(track_id, box, frame_index):
        measurement = OCSORT._box_to_measurement(box)
        state = torch.cat((measurement, measurement.new_zeros(3)))
        covariance = torch.diag(
            state.new_tensor([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])
        )
        observed_box = box.detach().to(dtype=torch.float64).clone()
        return {
            "id": track_id,
            "state": state,
            "covariance": covariance,
            "last_state": state.clone(),
            "last_covariance": covariance.clone(),
            "last_box": observed_box,
            "last_frame": frame_index,
            "history": [(frame_index, observed_box)],
            "time_since_update": 0,
            "hits": 1,
            "hit_streak": 1,
            "predicted_box": OCSORT._state_to_box(state),
        }

    def _update_track(self, track, box, frame_index):
        gap = frame_index - track["last_frame"]
        measurement = self._box_to_measurement(box)
        if gap > 1:
            state, covariance = self._oru_reupdate(
                track["last_state"],
                track["last_covariance"],
                track["last_box"],
                box.detach().to(dtype=torch.float64),
                gap - 1,
            )
            track["hit_streak"] = 0
        else:
            observation_matrix = self._measurement_matrix(track["state"])
            measurement_noise = self._measurement_noise(track["state"])
            state, covariance = self._kalman_update(
                track["state"], track["covariance"], measurement, observation_matrix, measurement_noise
            )
        observed_box = box.detach().to(dtype=torch.float64).clone()
        track["state"] = state
        track["covariance"] = covariance
        track["last_state"] = state.clone()
        track["last_covariance"] = covariance.clone()
        track["last_box"] = observed_box
        track["last_frame"] = frame_index
        track["history"].append((frame_index, observed_box))
        track["time_since_update"] = 0
        track["hits"] += 1
        track["hit_streak"] += 1
        track["predicted_box"] = self._state_to_box(state)

    def _validate_inputs(self, detections, detection_mask):
        if not isinstance(detections, torch.Tensor) or detections.ndim != 4 or detections.shape[-1] != 5:
            raise ValueError("detections must be a floating tensor with shape [B, T, N, 5]")
        if not detections.is_floating_point():
            raise ValueError("detections must use a floating point dtype")
        if detection_mask is None:
            return torch.ones(detections.shape[:-1], dtype=torch.bool, device=detections.device)
        if not isinstance(detection_mask, torch.Tensor) or detection_mask.shape != detections.shape[:-1]:
            raise ValueError("detection_mask must have shape [B, T, N]")
        if detection_mask.dtype != torch.bool or detection_mask.device != detections.device:
            raise ValueError("detection_mask must be a boolean tensor on the detections device")
        return detection_mask

    def forward(self, detections, detection_mask=None, return_track_boxes=False):
        if not self._is_boolean(return_track_boxes):
            raise ValueError("return_track_boxes must be a boolean")
        detection_mask = self._validate_inputs(detections, detection_mask)
        valid_detections = detections[detection_mask]
        if valid_detections.numel() > 0:
            if not torch.isfinite(valid_detections).all().item():
                raise ValueError("valid detections must have finite box coordinates and confidence")
            if ((valid_detections[:, 2] <= valid_detections[:, 0]) | (valid_detections[:, 3] <= valid_detections[:, 1])).any().item():
                raise ValueError("valid detections must have positive box width and height")
        batch_size, num_frames, _, _ = detections.shape
        track_boxes = torch.zeros_like(detections[..., :4])
        track_ids = torch.full(
            detections.shape[:-1], -1, dtype=torch.long, device=detections.device
        )
        track_mask = torch.zeros(detections.shape[:-1], dtype=torch.bool, device=detections.device)
        for batch_index in range(batch_size):
            tracks = []
            next_track_id = 1
            for frame_index in range(num_frames):
                frame = detections[batch_index, frame_index]
                valid_indices = torch.nonzero(detection_mask[batch_index, frame_index], as_tuple=False).flatten().tolist()
                high_confidence_indices = [
                    index
                    for index in valid_indices
                    if float(frame[index, 4].detach().item()) >= self.detection_threshold
                ]
                for track in tracks:
                    track["state"], track["covariance"] = self._kalman_predict(
                        track["state"], track["covariance"]
                    )
                    track["time_since_update"] += 1
                    track["predicted_box"] = self._state_to_box(track["state"])
                track_to_detection = {}
                primary = self._associate(frame, high_confidence_indices, tracks, True)
                matched_detection_indices = set()
                matched_track_indices = set()
                for detection_index, track_index in primary:
                    self._update_track(tracks[track_index], frame[detection_index, :4], frame_index)
                    track_to_detection[detection_index] = tracks[track_index]
                    matched_detection_indices.add(detection_index)
                    matched_track_indices.add(track_index)
                remaining_detections = [
                    index for index in high_confidence_indices if index not in matched_detection_indices
                ]
                remaining_tracks = [
                    track for index, track in enumerate(tracks) if index not in matched_track_indices
                ]
                ocr_pairs = self._associate(frame, remaining_detections, remaining_tracks, False)
                for detection_index, track_position in ocr_pairs:
                    track = remaining_tracks[track_position]
                    self._update_track(track, frame[detection_index, :4], frame_index)
                    track_to_detection[detection_index] = track
                    matched_detection_indices.add(detection_index)
                for detection_index in high_confidence_indices:
                    if detection_index in matched_detection_indices:
                        continue
                    track = self._new_track(
                        next_track_id,
                        frame[detection_index, :4].detach().to(dtype=torch.float64),
                        frame_index,
                    )
                    tracks.append(track)
                    track_to_detection[detection_index] = track
                    next_track_id += 1
                for detection_index, track in track_to_detection.items():
                    if track["hit_streak"] >= self.min_hits or frame_index < self.min_hits:
                        track_boxes[batch_index, frame_index, detection_index] = frame[detection_index, :4]
                        track_ids[batch_index, frame_index, detection_index] = track["id"]
                        track_mask[batch_index, frame_index, detection_index] = True
                tracks = [track for track in tracks if track["time_since_update"] <= self.max_age]
        if return_track_boxes:
            return track_boxes
        return track_boxes, track_ids, track_mask
