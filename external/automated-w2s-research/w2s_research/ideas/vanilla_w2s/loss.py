"""
Custom loss functions for vanilla weak-to-strong training.

Based on weak-to-strong's logconf_loss_fn approach.
"""
import torch
from transformers import Trainer


class LogconfLossTrainer(Trainer):
    """
    Custom Trainer that implements the logconf_loss_fn from weak-to-strong.

    This implements confidence-weighted loss that blends:
    1. Weak model's soft labels (pseudo-labels)
    2. Strong model's own predictions (self-supervision)

    The blending is controlled by:
    - aux_coef: How much to weigh strong model's predictions (default: 0.5)
    - warmup_frac: Fraction of training for warmup (default: 0.1)

    During training:
    - Early: Uses mostly weak labels (warmup phase)
    - Later: Blends weak labels with strong model's high-confidence predictions
    - Strong predictions are thresholded to match weak label distribution

    This approach helps the strong model go beyond weak supervision by
    gradually incorporating its own confident predictions.
    """

    def __init__(
        self,
        *args,
        label_token_ids=None,
        aux_coef=0.5,
        warmup_frac=0.2,
        **kwargs
    ):
        """
        Initialize SoftLossTrainer with logconf_loss_fn parameters.

        Args:
            label_token_ids: List of token IDs for label tokens (e.g., [" A", " B"])
            aux_coef: Coefficient for strong model predictions (default: 0.5)
            warmup_frac: Fraction of training steps for warmup (default: 0.1)
            *args, **kwargs: Standard Trainer arguments
        """
        super().__init__(*args, **kwargs)
        self.label_token_ids = label_token_ids
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute logconf_loss_fn - confidence-weighted loss with weak/strong blending.

        Implements the approach from weak-to-strong's logconf_loss_fn:
        1. Extract model logits for label token positions
        2. Compute strong model's predictions (softmax)
        3. Calculate threshold based on weak label distribution
        4. Create hard pseudo-labels from strong model using threshold
        5. Blend weak labels with strong predictions based on warmup schedule
        6. Compute cross-entropy with blended target

        Args:
            model: The model being trained
            inputs: Dictionary with keys:
                - input_ids: Token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - labels: Label IDs for loss (-100 for ignored positions) [batch_size, seq_len]
                - soft_labels: Soft label probabilities [batch_size, num_label_tokens]
            return_outputs: If True, return (loss, outputs). Otherwise return loss only.

        Returns:
            loss: Scalar loss tensor
            outputs: (optional) Model outputs
        """
        # Get model outputs
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        labels = inputs["labels"]  # Shape: [batch_size, seq_len]

        # Check if soft_labels are provided in the inputs
        if "soft_labels" not in inputs:
            # Fall back to standard cross-entropy with hard labels
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

        soft_labels = inputs["soft_labels"]  # Shape: [batch_size, num_label_tokens]

        # Convert to float tensors
        soft_labels = soft_labels.float()

        # Extract FULL vocabulary logits at label token positions
        # This ensures other tokens are penalized, not just label tokens
        batch_size = logits.size(0)
        vocab_size = logits.size(-1)
        batch_logits = []
        batch_label_probs = []  # Just the label token probs for blending logic

        for i in range(batch_size):
            # Find positions where we have labels (not -100)
            label_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]

            if len(label_positions) == 0:
                continue

            # get the token before the actual label token
            pos = label_positions[0] - 1

            # Get FULL vocabulary logits at this position
            position_logits = logits[i, pos]  # Shape: [vocab_size]

            batch_logits.append(position_logits)
            batch_label_probs.append(soft_labels[i])

        # Stack into tensors
        batch_logits = torch.stack(batch_logits).float()  # Shape: [batch_size, vocab_size]
        batch_label_probs = torch.stack(batch_label_probs).float()  # Shape: [batch_size, num_label_tokens]

        # Calculate warmup coefficient
        max_steps = self.state.max_steps if self.state.max_steps > 0 else 1
        step_frac = self.state.global_step / max_steps
        coef = 1.0 if step_frac > self.warmup_frac else step_frac / self.warmup_frac
        coef = coef * self.aux_coef

        # Get strong model predictions (only for label tokens, used for blending)
        label_logits = batch_logits[:, self.label_token_ids]  # Shape: [batch_size, num_label_tokens]
        preds = torch.softmax(label_logits, dim=-1)

        # Calculate mean weak labels to find class balance
        mean_weak = torch.mean(batch_label_probs, dim=0)
        assert mean_weak.shape == (2,), f"Expected 2 classes, got {mean_weak.shape}"

        # Find threshold that matches weak label distribution
        threshold = torch.quantile(preds[:, 0], mean_weak[1])

        # Create hard pseudo-labels from strong model (label tokens only)
        strong_label_preds = torch.cat(
            [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
            dim=1,
        ).float()

        # Blend weak labels with strong predictions (label tokens only)
        blended_label_probs = batch_label_probs * (1 - coef) + strong_label_preds.detach() * coef

        # Create full vocab target with blended label probs
        # All non-label tokens get 0 probability (penalized via softmax)
        target = torch.zeros(batch_logits.size(0), vocab_size, device=batch_logits.device, dtype=torch.float)
        for idx, token_id in enumerate(self.label_token_ids):
            target[:, token_id] = blended_label_probs[:, idx]

        # Compute cross-entropy with full vocab target
        # This penalizes probability mass on non-label tokens
        loss = torch.nn.functional.cross_entropy(
            batch_logits,
            target,
            reduction="mean",
        )

        return (loss, outputs) if return_outputs else loss
