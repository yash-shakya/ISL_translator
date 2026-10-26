
from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHECKPOINT_PATH = Path("checkpoints/isl_baseline_test.pt")
CACHE_PATH = Path("ISL_CSLRT_Corpus/processed/feature_cache.pt")

DEVICE = torch.device("cpu")
NUM_EXAMPLES = 15


# --------------------------------------------------
# 2. Model architecture (must match train.py)
# --------------------------------------------------

class SequenceHead(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, features):
        temporal_features, _ = self.lstm(features)
        return self.classifier(temporal_features)


# --------------------------------------------------
# 3. CTC greedy decoding
# --------------------------------------------------

def ctc_greedy_decode(logits, blank_id=0):
    """
    logits shape: [batch, time, classes]

    Collapse consecutive repeated predictions,
    then remove CTC blank tokens.
    """
    predicted_ids = logits.argmax(dim=-1)
    decoded_sequences = []

    for sequence in predicted_ids:
        tokens = []
        previous = None

        for token_id in sequence.tolist():
            if token_id != previous and token_id != blank_id:
                tokens.append(token_id)

            previous = token_id

        decoded_sequences.append(tokens)

    return decoded_sequences


# --------------------------------------------------
# 4. Convert token IDs to glosses
# --------------------------------------------------

def build_id_to_gloss(vocab):
    if isinstance(vocab, dict):
        return {int(index): token for token, index in vocab.items()}

    raise TypeError(
        f"Unexpected vocabulary format: {type(vocab)}"
    )


def decode_tokens(token_ids, id_to_gloss):
    return [
        id_to_gloss[token_id]
        for token_id in token_ids
        if token_id in id_to_gloss
    ]


# --------------------------------------------------
# 5. Word Error Rate
# --------------------------------------------------

def edit_distance(reference, hypothesis):
    previous_row = list(range(len(hypothesis) + 1))

    for i, ref_token in enumerate(reference, start=1):
        current_row = [i]

        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = (
                previous_row[j - 1]
                + (ref_token != hyp_token)
            )
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1

            current_row.append(
                min(substitution, insertion, deletion)
            )

        previous_row = current_row

    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    """
    WER = (substitutions + deletions + insertions)
          / number of reference words
    """
    if not reference:
        return 0.0 if not hypothesis else float("inf")

    return edit_distance(reference, hypothesis) / len(reference)


# --------------------------------------------------
# 6. Load checkpoint and cached validation features
# --------------------------------------------------

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not CACHE_PATH.exists():
        raise FileNotFoundError(CACHE_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    cache = torch.load(
        CACHE_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    vocab = checkpoint["vocab"]
    id_to_gloss = build_id_to_gloss(vocab)

    num_classes = max(id_to_gloss.keys()) + 1

    model = SequenceHead(
        num_classes=num_classes,
        hidden_size=128,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["head_state_dict"])
    model.eval()

    if "validation" not in cache:
        raise KeyError("Validation features are missing from the cache.")

    validation_batches = cache["validation"]

    total_edits = 0
    total_reference_tokens = 0
    exact_matches = 0
    total_sequences = 0
    examples = []

    # --------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------

    with torch.no_grad():
        for batch in validation_batches:
            features, flat_targets, target_lengths = batch

            features = features.to(DEVICE)
            flat_targets = flat_targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(features)
            predictions = ctc_greedy_decode(logits)

            # Split concatenated targets into per-video targets.
            offset = 0

            for i, predicted_ids in enumerate(predictions):
                target_length = int(target_lengths[i].item())

                reference_ids = flat_targets[
                    offset:offset + target_length
                ].tolist()

                offset += target_length

                reference = decode_tokens(
                    reference_ids, id_to_gloss
                )
                hypothesis = decode_tokens(
                    predicted_ids, id_to_gloss
                )

                edits = edit_distance(reference, hypothesis)

                total_edits += edits
                total_reference_tokens += len(reference)
                total_sequences += 1

                if reference == hypothesis:
                    exact_matches += 1

                if len(examples) < NUM_EXAMPLES:
                    examples.append((reference, hypothesis))

    # --------------------------------------------------
    # 8. Report results
    # --------------------------------------------------

    wer = (
        total_edits / total_reference_tokens
        if total_reference_tokens
        else 0.0
    )

    exact_match_rate = (
        exact_matches / total_sequences
        if total_sequences
        else 0.0
    )

    print("\n" + "=" * 55)
    print("VALIDATION RESULTS")
    print("=" * 55)

    print(f"Checkpoint epoch:       {checkpoint['epoch']}")
    print(f"Validation videos:      {total_sequences}")
    print(f"Validation loss saved:  {checkpoint['val_loss']:.4f}")
    print(f"Total reference glosses:{total_reference_tokens:6d}")
    print(f"Total edit operations:  {total_edits}")
    print(f"Word Error Rate (WER):  {wer:.2%}")
    print(f"Exact sequence match:   {exact_match_rate:.2%}")

    print("\n" + "=" * 55)
    print("SAMPLE PREDICTIONS")
    print("=" * 55)

    for index, (reference, hypothesis) in enumerate(
        examples, start=1
    ):
        print(f"\nExample {index}")
        print("Actual:   ", " ".join(reference))
        print("Predicted:", " ".join(hypothesis))


if __name__ == "__main__":
    main()
