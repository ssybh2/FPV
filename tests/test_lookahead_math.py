import math
import torch


def test_observation_preserves_v04_prefix_and_appends_lookahead():
    from q250_uzh.lookahead_racing_math import build_lookahead_observation

    current = torch.tensor([[1.0, 2.0, 3.0]])
    vel = torch.tensor([[4.0, 5.0, 6.0]])
    grav = torch.tensor([[7.0, 8.0, 9.0]])
    rates = torch.tensor([[10.0, 11.0, 12.0]])
    nxt = torch.tensor([[13.0, 14.0, 15.0]])
    ncur = torch.tensor([[1.0, 0.0, 0.0]])
    nnxt = torch.tensor([[0.0, 1.0, 0.0]])

    obs = build_lookahead_observation(current, vel, grav, rates, nxt, ncur, nnxt)
    assert obs.shape == (1, 21)
    assert torch.equal(obs[:, :12], torch.cat((current, vel, grav, rates), dim=-1))
    assert torch.equal(obs[:, 12:15], nxt)
    assert torch.equal(obs[:, 15:18], ncur)
    assert torch.equal(obs[:, 18:21], nnxt)


def test_gate_basis_is_orthonormal_and_normal_matches_yaw_pitch():
    from q250_uzh.lookahead_racing_math import gate_basis_from_yaw_pitch

    yaw = torch.tensor([math.radians(30.0)])
    pitch = torch.tensor([math.radians(10.0)])
    normal, right, up = gate_basis_from_yaw_pitch(yaw, pitch)

    expected = torch.tensor([[math.cos(math.radians(10))*math.cos(math.radians(30)),
                              math.cos(math.radians(10))*math.sin(math.radians(30)),
                              math.sin(math.radians(10))]])
    assert torch.allclose(normal, expected, atol=1e-6)
    assert torch.allclose((normal * right).sum(-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose((normal * up).sum(-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose((right * up).sum(-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(normal.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_gate_local_coordinates_detect_center_crossing_for_rotated_gate():
    from q250_uzh.lookahead_racing_math import gate_basis_from_yaw_pitch, gate_local_coordinates

    yaw = torch.tensor([math.radians(25.0)])
    pitch = torch.tensor([math.radians(-8.0)])
    normal, right, up = gate_basis_from_yaw_pitch(yaw, pitch)
    center = torch.tensor([[3.0, 1.0, 1.5]])
    position = center + 0.2 * normal + 0.3 * right - 0.1 * up
    signed, lateral, vertical = gate_local_coordinates(position, center, normal, right, up)
    assert torch.allclose(signed, torch.tensor([0.2]), atol=1e-6)
    assert torch.allclose(lateral, torch.tensor([0.3]), atol=1e-6)
    assert torch.allclose(vertical, torch.tensor([-0.1]), atol=1e-6)


def test_curriculum_progresses_to_five_gate_oriented_racing():
    from q250_uzh.lookahead_racing_math import lookahead_curriculum

    s0 = lookahead_curriculum(0)
    s1 = lookahead_curriculum(1200)
    s2 = lookahead_curriculum(5000)
    assert (s0.stage, s0.gate_count) == (0, 3)
    assert (s1.stage, s1.gate_count) == (1, 3)
    assert (s2.stage, s2.gate_count) == (2, 5)
    assert s0.yaw_jitter_deg == 0.0
    assert s1.yaw_jitter_deg > 0.0
    assert s2.yaw_jitter_deg > s1.yaw_jitter_deg


def test_gate_quaternion_rotates_local_x_to_gate_normal():
    from q250_uzh.lookahead_racing_math import gate_basis_from_yaw_pitch, gate_quat_wxyz_from_yaw_pitch

    yaw = torch.tensor([math.radians(-35.0)])
    pitch = torch.tensor([math.radians(12.0)])
    normal, _, _ = gate_basis_from_yaw_pitch(yaw, pitch)
    q = gate_quat_wxyz_from_yaw_pitch(yaw, pitch)
    # Rotate local +X into world using q v q^-1.
    w = q[:, :1]
    qv = q[:, 1:]
    x = torch.tensor([[1.0, 0.0, 0.0]])
    t = 2.0 * torch.cross(qv, x, dim=-1)
    rotated = x + w * t + torch.cross(qv, t, dim=-1)
    assert torch.allclose(rotated, normal, atol=1e-6)
