import math

import torch
from torch import nn
from torch.nn import functional as F


class BigGAN(nn.Module):
    architecture_id = "biggan"
    PRESETS = {
        "tiny": {
            "image_size": 128,
            "channel_width": 8,
            "num_classes": 10,
            "latent_dim": 120,
            "shared_embedding_dim": 128,
        },
        "small": {
            "image_size": 128,
            "channel_width": 44,
            "num_classes": 1000,
            "latent_dim": 120,
            "shared_embedding_dim": 128,
        },
        "base": {
            "image_size": 128,
            "channel_width": 96,
            "num_classes": 1000,
            "latent_dim": 120,
            "shared_embedding_dim": 128,
        },
    }

    def __init__(
        self,
        image_size=128,
        channel_width=96,
        num_classes=1000,
        latent_dim=None,
        shared_embedding_dim=128,
        truncation_threshold=None,
    ):
        super().__init__()
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size not in (128, 256, 512):
            raise ValueError("image_size must be one of 128, 256, or 512")
        if isinstance(channel_width, bool) or not isinstance(channel_width, int) or channel_width <= 0:
            raise ValueError("channel_width must be a positive integer")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(shared_embedding_dim, bool) or not isinstance(shared_embedding_dim, int) or shared_embedding_dim <= 0:
            raise ValueError("shared_embedding_dim must be a positive integer")
        latent_sizes = {128: 120, 256: 140, 512: 160}
        expected_latent_dim = latent_sizes[image_size]
        if latent_dim is None:
            latent_dim = expected_latent_dim
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim != expected_latent_dim:
            raise ValueError(f"latent_dim must be {expected_latent_dim} for image_size {image_size}")
        if truncation_threshold is not None:
            if isinstance(truncation_threshold, bool) or not isinstance(truncation_threshold, (int, float)):
                raise ValueError("truncation_threshold must be a positive finite number or None")
            if not math.isfinite(truncation_threshold) or truncation_threshold <= 0:
                raise ValueError("truncation_threshold must be a positive finite number or None")

        self.image_size = image_size
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.shared_embedding_dim = shared_embedding_dim
        self.chunk_width = 20
        self.truncation_threshold = truncation_threshold
        self.num_blocks = int(math.log2(image_size)) - 2
        self.condition_dim = shared_embedding_dim + self.chunk_width
        width = channel_width
        if image_size == 128:
            generator_multipliers = [16, 8, 4, 2, 1]
            discriminator_multipliers = [2, 4, 8, 16, 16]
            self.generator_attention_resolution = 64
        elif image_size == 256:
            generator_multipliers = [16, 8, 8, 4, 2, 1]
            discriminator_multipliers = [2, 4, 8, 8, 16, 16]
            self.generator_attention_resolution = 128
        else:
            generator_multipliers = [16, 8, 8, 4, 4, 2, 1]
            discriminator_multipliers = [2, 4, 4, 8, 8, 16, 16]
            self.generator_attention_resolution = 64

        self.class_embedding = nn.Embedding(num_classes, shared_embedding_dim)
        self.generator_input = self._spectral_norm(
            nn.Linear(self.condition_dim, 16 * width * 4 * 4)
        )
        self.generator_channels = [multiplier * width for multiplier in generator_multipliers]
        generator_inputs = [16 * width, *self.generator_channels[:-1]]
        self.generator_norm1 = nn.ModuleList()
        self.generator_norm2 = nn.ModuleList()
        self.generator_condition1 = nn.ModuleList()
        self.generator_condition2 = nn.ModuleList()
        self.generator_conv1 = nn.ModuleList()
        self.generator_conv2 = nn.ModuleList()
        self.generator_skip = nn.ModuleList()
        for in_channels, out_channels in zip(generator_inputs, self.generator_channels):
            self.generator_norm1.append(nn.BatchNorm2d(in_channels, affine=False))
            self.generator_norm2.append(nn.BatchNorm2d(out_channels, affine=False))
            self.generator_condition1.append(
                self._condition_projection(self.condition_dim, in_channels)
            )
            self.generator_condition2.append(
                self._condition_projection(self.condition_dim, out_channels)
            )
            self.generator_conv1.append(
                self._spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            )
            self.generator_conv2.append(
                self._spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            )
            self.generator_skip.append(
                nn.Identity()
                if in_channels == out_channels
                else self._spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            )
        final_channels = self.generator_channels[-1]
        self.generator_final_norm = nn.BatchNorm2d(final_channels)
        self.generator_final_conv = self._spectral_norm(nn.Conv2d(final_channels, 3, 3, padding=1))

        self.generator_attention = self._attention_layers(self.generator_channels[3 if image_size == 128 else 4 if image_size == 256 else 3])
        self.generator_attention_gamma = nn.Parameter(torch.zeros(()))

        discriminator_inputs = [3, *[multiplier * width for multiplier in discriminator_multipliers[:-1]]]
        self.discriminator_channels = [multiplier * width for multiplier in discriminator_multipliers]
        self.discriminator_conv1 = nn.ModuleList()
        self.discriminator_conv2 = nn.ModuleList()
        self.discriminator_skip = nn.ModuleList()
        for in_channels, out_channels in zip(discriminator_inputs, self.discriminator_channels):
            self.discriminator_conv1.append(
                self._spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            )
            self.discriminator_conv2.append(
                self._spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            )
            self.discriminator_skip.append(
                self._spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            )
        final_discriminator_channels = self.discriminator_channels[-1]
        self.discriminator_final_conv1 = self._spectral_norm(
            nn.Conv2d(final_discriminator_channels, final_discriminator_channels, 3, padding=1)
        )
        self.discriminator_final_conv2 = self._spectral_norm(
            nn.Conv2d(final_discriminator_channels, final_discriminator_channels, 3, padding=1)
        )
        self.discriminator_attention = self._attention_layers(
            self.discriminator_channels[0 if image_size == 128 else 1 if image_size == 256 else 2]
        )
        self.discriminator_attention_gamma = nn.Parameter(torch.zeros(()))
        self.discriminator_class_embedding = nn.Embedding(num_classes, final_discriminator_channels)
        self.discriminator_linear = self._spectral_norm(nn.Linear(final_discriminator_channels, 1))

    @staticmethod
    def _spectral_norm(module):
        return nn.utils.parametrizations.spectral_norm(module)

    @classmethod
    def _condition_projection(cls, condition_dim, channels):
        projection = nn.Linear(condition_dim, channels * 2)
        with torch.no_grad():
            projection.weight.zero_()
            if projection.bias is not None:
                projection.bias.zero_()
                projection.bias[:channels].fill_(1)
        return projection

    @classmethod
    def _attention_layers(cls, channels):
        query_channels = max(channels // 8, 1)
        value_channels = max(channels // 2, 1)
        return nn.ModuleDict(
            {
                "query": cls._spectral_norm(nn.Conv2d(channels, query_channels, 1, bias=False)),
                "key": cls._spectral_norm(nn.Conv2d(channels, query_channels, 1, bias=False)),
                "value": cls._spectral_norm(nn.Conv2d(channels, value_channels, 1, bias=False)),
                "output": cls._spectral_norm(nn.Conv2d(value_channels, channels, 1, bias=False)),
            }
        )

    @staticmethod
    def split_latent(z, chunk_width=20):
        if not isinstance(z, torch.Tensor) or z.ndim != 2:
            raise ValueError("z must have shape (batch, latent_dim)")
        if isinstance(chunk_width, bool) or not isinstance(chunk_width, int) or chunk_width <= 0:
            raise ValueError("chunk_width must be a positive integer")
        if z.shape[1] == 0 or z.shape[1] % chunk_width:
            raise ValueError("latent width must be divisible by chunk_width")
        return tuple(z.split(chunk_width, dim=1))

    @classmethod
    def hierarchical_conditioning(cls, z, class_embedding, chunk_width=20):
        if not isinstance(class_embedding, torch.Tensor) or class_embedding.ndim != 2:
            raise ValueError("class_embedding must have shape (batch, embedding_dim)")
        if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] != class_embedding.shape[0]:
            raise ValueError("z and class_embedding must have matching batch dimensions")
        return tuple(
            torch.cat((chunk, class_embedding), dim=1)
            for chunk in cls.split_latent(z, chunk_width)
        )

    @staticmethod
    def conditional_batch_norm(x, mean, variance, scale, bias, epsilon=1e-5):
        if not isinstance(x, torch.Tensor) or x.ndim < 2:
            raise ValueError("x must have a batch and channel dimension")
        if not all(isinstance(value, torch.Tensor) for value in (mean, variance, scale, bias)):
            raise ValueError("mean, variance, scale, and bias must be tensors")
        channels = x.shape[1]

        def channel_view(value):
            if value.ndim == 1:
                if value.shape[0] != channels:
                    raise ValueError("normalization parameters must match the channel dimension")
                return value.reshape(1, channels, *([1] * (x.ndim - 2)))
            if value.ndim == 2 and value.shape == (x.shape[0], channels):
                return value.reshape(x.shape[0], channels, *([1] * (x.ndim - 2)))
            return value

        if not isinstance(epsilon, (int, float)) or epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a non-negative finite number")
        return (x - channel_view(mean)) / torch.sqrt(channel_view(variance) + epsilon) * channel_view(scale) + channel_view(bias)

    @staticmethod
    def hinge_losses(real_scores, fake_scores):
        if not isinstance(real_scores, torch.Tensor) or not isinstance(fake_scores, torch.Tensor):
            raise TypeError("real_scores and fake_scores must be tensors")
        if real_scores.numel() == 0 or fake_scores.numel() == 0:
            raise ValueError("real_scores and fake_scores must be non-empty")
        discriminator_loss = F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()
        generator_loss = -fake_scores.mean()
        return discriminator_loss, generator_loss

    @staticmethod
    def modified_orthogonal_regularizer(weight, beta=1e-4):
        if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
            raise ValueError("weight must be a tensor with at least two dimensions")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or beta < 0 or not math.isfinite(beta):
            raise ValueError("beta must be a non-negative finite number")
        matrix = weight.reshape(weight.shape[0], -1)
        gram = matrix.transpose(0, 1) @ matrix
        off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
        return beta * off_diagonal.square().sum()

    @staticmethod
    def projection_score_components(features, unconditional_weights, class_embedding, bias=0.0):
        if not all(isinstance(value, torch.Tensor) for value in (features, unconditional_weights, class_embedding)):
            raise ValueError("features, unconditional_weights, and class_embedding must be tensors")
        if features.ndim != 2 or unconditional_weights.ndim != 1 or class_embedding.ndim not in (1, 2):
            raise ValueError("projection score inputs must be a feature matrix and matching weight vectors")
        if features.shape[1] != unconditional_weights.shape[0]:
            raise ValueError("projection score weights must match the feature width")
        if class_embedding.ndim == 1:
            if class_embedding.shape[0] != features.shape[1]:
                raise ValueError("projection score weights must match the feature width")
        elif class_embedding.shape != features.shape:
            raise ValueError("batched class embeddings must match the feature matrix shape")
        return (features * unconditional_weights).sum(dim=1, keepdim=True) + (features * class_embedding).sum(dim=1, keepdim=True) + bias

    def sample_latent(self, batch_size, device=None, dtype=None, truncation_threshold=None, generator=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        threshold = self.truncation_threshold if truncation_threshold is None else truncation_threshold
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("truncation_threshold must be a positive finite number or None")
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("truncation_threshold must be a positive finite number or None")
        parameter = self.class_embedding.weight
        sample_device = parameter.device if device is None else torch.device(device)
        sample_dtype = parameter.dtype if dtype is None else dtype
        latent = torch.randn(batch_size, self.latent_dim, device=sample_device, dtype=sample_dtype, generator=generator)
        if threshold is None:
            return latent
        invalid = latent.abs() > threshold
        while bool(invalid.any()):
            replacement = torch.randn(
                batch_size,
                self.latent_dim,
                device=sample_device,
                dtype=sample_dtype,
                generator=generator,
            )
            latent = torch.where(invalid, replacement, latent)
            invalid = latent.abs() > threshold
        return latent

    def _validate_labels(self, labels, batch_size):
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("labels must use an integer dtype")
        if labels.numel() and (bool((labels < 0).any()) or bool((labels >= self.num_classes).any())):
            raise ValueError("labels must be in the range [0, num_classes)")

    def _validate_latent(self, z, labels):
        if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape (batch, {self.latent_dim})")
        if not z.is_floating_point():
            raise ValueError("z must use a floating point dtype")
        if z.shape[0] <= 0:
            raise ValueError("z batch must be non-empty")
        self._validate_labels(labels, z.shape[0])

    def _conditioned_norm(self, x, norm, projection, condition):
        normalized = norm(x)
        parameters = projection(condition)
        scale, bias = parameters.chunk(2, dim=1)
        zeros = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        ones = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
        return self.conditional_batch_norm(normalized, zeros, ones, scale, bias, 0.0)

    def _generator_block(self, x, condition, index):
        hidden = self._conditioned_norm(
            x, self.generator_norm1[index], self.generator_condition1[index], condition
        )
        hidden = F.interpolate(F.relu(hidden), scale_factor=2, mode="nearest")
        hidden = self.generator_conv1[index](hidden)
        hidden = self._conditioned_norm(
            hidden, self.generator_norm2[index], self.generator_condition2[index], condition
        )
        hidden = self.generator_conv2[index](F.relu(hidden))
        skip = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.generator_skip[index](skip)
        return hidden + skip

    def _apply_attention(self, x, layers, gamma):
        batch, _, height, width = x.shape
        query = layers["query"](x).flatten(2).transpose(1, 2)
        pooled = F.max_pool2d(x, kernel_size=2, stride=2)
        key = layers["key"](pooled).flatten(2)
        value = layers["value"](pooled).flatten(2)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        attended = torch.bmm(value, attention.transpose(1, 2)).reshape(
            batch, -1, height, width
        )
        return x + gamma * layers["output"](attended)

    def non_local_attention(self, x, network="generator"):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width)")
        if network == "generator":
            layers = self.generator_attention
            gamma = self.generator_attention_gamma
        elif network == "discriminator":
            layers = self.discriminator_attention
            gamma = self.discriminator_attention_gamma
        else:
            raise ValueError("network must be 'generator' or 'discriminator'")
        expected_channels = layers["query"].in_channels
        if x.shape[1] != expected_channels or x.shape[0] <= 0:
            raise ValueError("x channel count must match the selected attention block")
        return self._apply_attention(x, layers, gamma)

    def forward(self, z, labels):
        self._validate_latent(z, labels)
        embedding = self.class_embedding(labels)
        conditions = self.hierarchical_conditioning(z, embedding, self.chunk_width)
        x = self.generator_input(conditions[0]).reshape(z.shape[0], 16 * self.channel_width, 4, 4)
        resolution = 4
        for index, condition in enumerate(conditions[1:]):
            x = self._generator_block(x, condition, index)
            resolution *= 2
            if resolution == self.generator_attention_resolution:
                x = self._apply_attention(x, self.generator_attention, self.generator_attention_gamma)
        x = F.relu(self.generator_final_norm(x))
        return torch.tanh(self.generator_final_conv(x))

    def discriminator(self, images, labels):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape (batch, 3, image_size, image_size)")
        if images.shape[0] <= 0 or images.shape[1] != 3 or images.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"images must have shape (batch, 3, {self.image_size}, {self.image_size})")
        self._validate_labels(labels, images.shape[0])
        x = images
        resolution = self.image_size
        for index, (conv1, conv2, skip) in enumerate(
            zip(self.discriminator_conv1, self.discriminator_conv2, self.discriminator_skip)
        ):
            hidden = conv1(F.relu(x))
            hidden = conv2(F.relu(hidden))
            hidden = F.avg_pool2d(hidden, kernel_size=2, stride=2)
            shortcut = F.avg_pool2d(skip(x), kernel_size=2, stride=2)
            x = hidden + shortcut
            resolution //= 2
            if resolution == 64:
                x = self._apply_attention(
                    x, self.discriminator_attention, self.discriminator_attention_gamma
                )
        hidden = self.discriminator_final_conv1(F.relu(x))
        hidden = self.discriminator_final_conv2(F.relu(hidden))
        x = hidden + x
        features = F.relu(x).sum(dim=(2, 3))
        return self.projection_score(features, labels)

    def projection_score(self, features, labels):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must have shape (batch, feature_dim)")
        if features.shape[1] != self.discriminator_linear.in_features:
            raise ValueError("features must match the discriminator feature width")
        self._validate_labels(labels, features.shape[0])
        class_features = self.discriminator_class_embedding(labels)
        weights = self.discriminator_linear.weight.reshape(-1)
        return self.projection_score_components(
            features, weights, class_features
        ) + self.discriminator_linear.bias
