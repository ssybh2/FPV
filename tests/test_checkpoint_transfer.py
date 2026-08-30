import torch


def test_transfer_expands_12d_input_layers_and_zeros_new_columns():
    from q250_uzh.checkpoint_transfer import expand_state_dict_observation_input

    old = {
        "actor.0.weight": torch.arange(128 * 12, dtype=torch.float32).reshape(128, 12),
        "actor.0.bias": torch.ones(128),
        "critic.0.weight": torch.full((128, 12), 2.0),
        "critic.0.bias": torch.ones(128) * 3,
        "actor.2.weight": torch.full((128, 128), 4.0),
        "std": torch.ones(4) * 9.0,
    }
    new = {
        "actor.0.weight": torch.randn(128, 21),
        "actor.0.bias": torch.zeros(128),
        "critic.0.weight": torch.randn(128, 21),
        "critic.0.bias": torch.zeros(128),
        "actor.2.weight": torch.zeros(128, 128),
        "std": torch.ones(4) * 0.35,
    }

    merged, report = expand_state_dict_observation_input(old, new, old_obs_dim=12, new_obs_dim=21)
    assert torch.equal(merged["actor.0.weight"][:, :12], old["actor.0.weight"])
    assert torch.count_nonzero(merged["actor.0.weight"][:, 12:]) == 0
    assert torch.equal(merged["critic.0.weight"][:, :12], old["critic.0.weight"])
    assert torch.count_nonzero(merged["critic.0.weight"][:, 12:]) == 0
    assert torch.equal(merged["actor.2.weight"], old["actor.2.weight"])
    assert torch.equal(merged["actor.0.bias"], old["actor.0.bias"])
    # Deliberately keep v0.5 exploration noise instead of the late v0.4 std.
    assert torch.equal(merged["std"], new["std"])
    assert report.expanded_input_layers == 2
    assert report.copied_exact >= 3


def test_extract_model_state_dict_accepts_rsl_rl_checkpoint_shape():
    from q250_uzh.checkpoint_transfer import extract_model_state_dict

    state = {"actor.0.weight": torch.ones(2, 12)}
    ckpt = {"model_state_dict": state, "optimizer_state_dict": {"x": 1}, "iter": 399}
    extracted = extract_model_state_dict(ckpt)
    assert extracted is state


def test_save_transfer_snapshot_does_not_require_runner_logger_state(tmp_path):
    """Pre-training transfer snapshots must not call OnPolicyRunner.save()."""
    from types import SimpleNamespace
    from q250_uzh.checkpoint_transfer import save_transfer_snapshot

    policy = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer),
        current_learning_iteration=0,
    )
    # Intentionally no runner.logger_type / runner.writer: these are created by learn().
    path = tmp_path / "model_transfer_init.pt"
    save_transfer_snapshot(runner, path, source="model_399.pt")

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert path.exists()
    assert saved["iter"] == 0
    assert saved["infos"]["source"] == "model_399.pt"
    assert set(saved["model_state_dict"]) == set(policy.state_dict())
    assert "optimizer_state_dict" in saved
