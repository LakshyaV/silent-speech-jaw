from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ssj.data import AugmentConfig, FeatureDataset
from ssj.data.loader import BucketSampler, UtteranceDataset, collate
from ssj.decode.ctc_score import ctc_forward, log_softmax
from ssj.dsp import N_FEATURES
from ssj.models import (
    ExportWrapper,
    JawPhonemeNet,
    ModelConfig,
    MultiResolutionCTC,
    greedy_decode_batch,
    output_lengths,
)
from ssj.models.encoder import BiGRU, DepthwiseConv1d
from ssj.runtime.bundle import Bundle, BundleError
from ssj.train.trainer import (
    ExperimentConfig,
    TrainConfig,
    load_checkpoint,
    predict,
    score_predictions,
    train,
    warm_start,
)

SMALL = ModelConfig(stem_channels=8, d_model=32, tcn_dilations=(1, 2), gru_layers=1, gru_hidden=24, dropout=0.0)


def test_depthwise_convolution_matches_reference():
    torch.manual_seed(0)
    for kernel, dilation in [(5, 1), (5, 4), (3, 8)]:
        conv = DepthwiseConv1d(16, kernel, dilation)
        x = torch.randn(3, 16, 50)
        reference = F.conv1d(x, conv.weight, None, 1, conv.padding, conv.dilation, 16)
        assert torch.allclose(conv(x), reference, atol=1e-5)
        conv.native = True
        assert torch.allclose(conv(x), reference, atol=1e-6)


def test_bigru_ignores_padding():
    torch.manual_seed(0)
    gru = BiGRU(8, 6, 2, 0.0).eval()
    x = torch.randn(3, 20, 8)
    lengths = torch.tensor([20, 13, 7])
    batched = gru(x, lengths)
    for i, n in enumerate(lengths):
        assert torch.allclose(batched[i, :n], gru(x[i : i + 1, :n])[0], atol=1e-5)


def test_forward_shapes_and_padding_invariance(inventory):
    torch.manual_seed(0)
    model = JawPhonemeNet(SMALL, inventory.size, n_sessions=2).eval()
    x = torch.randn(3, 101, N_FEATURES)
    lengths = torch.tensor([101, 60, 33])
    out = model(x, lengths, torch.tensor([0, 1, 2]))
    assert out["logits"].shape == (3, 51, inventory.size)
    assert out["lengths"].tolist() == [51, 30, 17] == output_lengths(lengths, 2).tolist()
    assert out["inter_logits"] is None  # only produced in training
    for i, n in enumerate(lengths):  # an utterance's output does not depend on what it is batched with
        alone = model(x[i : i + 1, :n], lengths[i : i + 1], torch.tensor([i]))
        assert torch.allclose(out["logits"][i, : out["lengths"][i]], alone["logits"][0], atol=1e-4)
    assert 10_000 < model.parameter_count() < 60_000
    assert 500_000 < JawPhonemeNet(ModelConfig(), inventory.size).parameter_count() < 1_000_000


def test_session_adapters(inventory):
    torch.manual_seed(0)
    model = JawPhonemeNet(SMALL, inventory.size, n_sessions=3).eval()
    x = torch.randn(2, 40, N_FEATURES)
    assert torch.equal(model.adapt(x, torch.tensor([0, 2])), x)  # identity at initialisation
    assert float(model.adapter_penalty().detach()) == 0.0
    with torch.no_grad():
        model.adapter_delta[2] = 0.1 * torch.eye(N_FEATURES)
        model.adapter_bias[2] = 1.0
    adapted = model.adapt(x, torch.tensor([0, 2]))
    assert torch.equal(adapted[0], x[0]) and torch.allclose(adapted[1], 1.1 * x[1] + 1.0, atol=1e-5)
    assert float(model.adapter_penalty().detach()) > 0
    baked = ExportWrapper(model, session_id=2)
    with torch.no_grad():
        expected = torch.log_softmax(model(x[1:2], session=torch.tensor([2]))["logits"], -1)
        assert torch.allclose(baked(x[1:2]), expected, atol=1e-4)
    assert JawPhonemeNet(SMALL, inventory.size, n_sessions=0).adapter_delta is None
    with pytest.raises(ValueError):
        ModelConfig(streams=("jaw", "tongue"))
    assert JawPhonemeNet(
        ModelConfig(streams=("rel",), stem_channels=8, d_model=32, tcn_dilations=(1,), gru_layers=0), inventory.size
    )(x)["logits"].shape == (2, 20, inventory.size)


def test_multi_resolution_loss(inventory):
    torch.manual_seed(0)
    criterion = MultiResolutionCTC(inventory, coarse_weight=0.3, inter_weight=0.0)
    logits = torch.randn(2, 30, inventory.size, requires_grad=True)
    log_probs = torch.log_softmax(logits, -1)
    classes = criterion.marginalize(log_probs)
    assert classes.shape == (2, 30, inventory.n_classes)
    assert torch.allclose(classes.exp().sum(-1), torch.ones(2, 30), atol=1e-5)

    targets = torch.tensor([inventory.encode(["P", "AE", "T"]), inventory.encode(["B", "AE", "D"])])
    lengths, target_lengths = torch.tensor([30, 30]), torch.tensor([3, 3])
    loss, parts = criterion(logits, lengths, targets, target_lengths)
    assert set(parts) == {"phoneme", "class"} and parts["class"] <= parts["phoneme"] + 1e-5  # the class task is easier
    reference = -ctc_forward(log_softmax(logits[0].detach().numpy().astype(np.float64)), targets[0].tolist())
    fine_only, _ = MultiResolutionCTC(inventory, 0.0, 0.0)(logits[:1], lengths[:1], targets[:1], target_lengths[:1])
    # ... and agrees with the independent numpy forward algorithm
    assert float(fine_only.detach()) * 3 == pytest.approx(reference, rel=1e-4)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    # Confusing P with B costs phoneme loss but not class loss.
    confident = torch.full((1, 9, inventory.size), -20.0)
    for frame, symbol in zip(range(1, 9, 3), ["B", "AE", "T"]):
        confident[0, frame, inventory.token_id(symbol)] = 20.0
    confident[0, [0, 2, 3, 5, 6, 8], 0] = 20.0
    _, parts = criterion(confident, torch.tensor([9]), targets[:1], torch.tensor([3]))
    assert parts["class"] < 0.01 < 1.0 < parts["phoneme"]
    assert greedy_decode_batch(torch.log_softmax(confident, -1), [9]) == [inventory.encode(["B", "AE", "T"])]
    with pytest.raises(ValueError):
        MultiResolutionCTC(inventory, coarse_weight=1.5)


def test_loader_batches(dataset_dir):
    data = FeatureDataset(dataset_dir)
    items = data.split("train")
    index = {"sim/s00": 1}
    dataset = UtteranceDataset(data, items, index, augment_config=AugmentConfig(concat_prob=1.0), seed=0)
    sample = dataset[0]
    assert sample["x"].shape[1] == N_FEATURES and sample["session"] == 1
    assert len(sample["y"]) > len(items[0].phonemes)  # two utterances were joined
    plain = UtteranceDataset(data, items, index)
    assert plain[0]["y"].tolist() == data.inventory.encode(items[0].phonemes)
    shuffled = UtteranceDataset(data, items, index, label_permutation=list(range(len(items)))[::-1])
    assert shuffled[0]["y"].tolist() == data.inventory.encode(items[-1].phonemes)

    batch = collate([plain[i] for i in range(4)])
    assert batch["x"].shape == (4, int(batch["lengths"].max()), N_FEATURES)
    assert batch["y"].shape == (4, int(batch["target_lengths"].max()))
    for row, n in enumerate(batch["lengths"]):
        assert torch.all(batch["x"][row, n:] == 0)

    sampler = BucketSampler(plain.lengths, 8, seed=0, curriculum_epochs=1)
    first = [i for batch in sampler for i in batch]
    assert [plain.lengths[i] for i in first] == sorted(plain.lengths)  # first epoch: shortest first
    second = [i for batch in sampler for i in batch]
    assert sorted(second) == list(range(len(items))) and second != first
    assert len(sampler) == 4


def test_training_run_artifacts(run_dir, dataset_dir, inventory):
    assert {"checkpoint.pt", "config.yaml", "history.csv", "metrics.json"} <= {p.name for p in run_dir.iterdir()}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["shuffled_labels"] is False and 0 < metrics["val"]["per"] <= 1.5 and "test" in metrics
    history = (run_dir / "history.csv").read_text().splitlines()
    assert history[0].startswith("epoch,train_loss,val_per") and len(history) == 4
    losses = [float(line.split(",")[1]) for line in history[1:]]
    assert losses[-1] < losses[0]  # it is learning

    model, payload = load_checkpoint(run_dir / "checkpoint.pt")
    assert (
        payload["sessions"] == ["sim/s00"]
        and payload["frontend_fingerprint"] == FeatureDataset(dataset_dir).meta["frontend_fingerprint"]
    )
    data = FeatureDataset(dataset_dir)
    items = data.split("test")[:6]
    log_probs = predict(model, data, items)
    assert all(lp.shape == ((i.n_frames + 1) // 2, inventory.size) for lp, i in zip(log_probs, items))
    assert all(np.allclose(np.exp(lp).sum(-1), 1.0, atol=1e-4) for lp in log_probs)
    scores = score_predictions(log_probs, items, inventory)
    assert set(scores) == {"per", "class_er", "utterances"}

    fresh = JawPhonemeNet(model.config, inventory.size, n_sessions=5)
    assert warm_start(fresh, run_dir / "checkpoint.pt") == []
    assert torch.equal(fresh.head.weight, model.head.weight) and fresh.adapter_delta.shape[0] == 6


def test_shuffled_label_control_runs(dataset_dir, tmp_path):
    config = ExperimentConfig(
        model=SMALL, train=TrainConfig(epochs=1, batch_size=12, warmup_steps=2, shuffle_labels=True, threads=1)
    )
    messages: list[str] = []
    metrics = train(dataset_dir, tmp_path / "control", config, report=messages.append)
    assert metrics["shuffled_labels"] is True and any("CONTROL RUN" in m for m in messages)


def test_export_and_bundle(bundle_dir, run_dir, dataset_dir, inventory):
    meta = json.loads((bundle_dir / "bundle.json").read_text())
    assert meta["model"]["max_export_error"] < 1e-3 and meta["model"]["bytes"] > 10_000
    assert (bundle_dir / "model.onnx").exists()

    bundle = Bundle.load(bundle_dir)
    assert bundle.inventory == inventory and bundle.frame_rate == 50.0
    data = FeatureDataset(dataset_dir)
    model, _ = load_checkpoint(run_dir / "checkpoint.pt")
    for item in data.split("test")[:4]:  # the runtime reproduces the training-time model
        reference = predict(model, data, [item])[0]
        produced = bundle.log_probs(data.load(item))
        assert produced.shape == reference.shape and np.abs(produced - reference).max() < 1e-3
    assert bundle.log_probs(np.zeros((0, N_FEATURES), dtype=np.float32)).shape == (0, inventory.size)
    assert "parameters" in bundle.describe()


def test_bundle_refuses_a_foreign_frontend(bundle_dir, tmp_path):
    import shutil

    broken = tmp_path / "bundle"
    shutil.copytree(bundle_dir, broken)
    meta = json.loads((broken / "bundle.json").read_text())
    meta["frontend"]["lowpass_hz"] = 12.0  # features would be computed differently from training
    (broken / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(BundleError, match="front-end mismatch"):
        Bundle.load(broken)
    with pytest.raises(BundleError, match="not a bundle"):
        Bundle.load(tmp_path)
