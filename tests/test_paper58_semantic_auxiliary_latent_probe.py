import numpy as np


def test_torch_decoder_head_matches_sklearn_multiclass_logits() -> None:
    import torch

    from scripts.paper58_benchmark.train_paper58_semantic_auxiliary_latent_probe import (
        build_torch_decoder_head,
    )

    class ToyDecoder:
        classes_ = np.array([1, 5, 11], dtype=np.int32)
        coef_ = np.array(
            [
                [0.2, -0.1],
                [-0.3, 0.4],
                [0.1, 0.2],
            ],
            dtype=np.float32,
        )
        intercept_ = np.array([0.5, -0.2, 0.1], dtype=np.float32)

    head = build_torch_decoder_head(ToyDecoder(), device=torch.device("cpu"))
    features = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32)

    logits = head(features).detach().numpy().reshape(3)
    expected = ToyDecoder.coef_ @ np.array([1.0, 2.0], dtype=np.float32) + ToyDecoder.intercept_

    assert np.allclose(logits, expected, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in head.parameters())


def test_labels_to_decoder_indices_ignores_unknown_classes() -> None:
    import torch

    from scripts.paper58_benchmark.train_paper58_semantic_auxiliary_latent_probe import (
        labels_to_decoder_indices,
    )

    labels = np.array([[1, 5, 8], [11, 0, 5]], dtype=np.int32)
    indices = labels_to_decoder_indices(
        labels,
        decoder_classes=[1, 5, 11],
        ignore_index=-100,
        device=torch.device("cpu"),
    )

    assert indices.tolist() == [[0, 1, -100], [2, -100, 1]]
    assert indices.dtype == torch.long


def test_decoder_change_logits_compare_start_against_other_classes() -> None:
    import torch

    from scripts.paper58_benchmark.train_paper58_semantic_auxiliary_latent_probe import (
        decoder_change_logits,
    )

    logits = torch.tensor(
        [
            [
                [[4.0, 0.0]],  # class 1
                [[0.0, 3.0]],  # class 5
                [[-1.0, 1.0]],  # class 11
            ]
        ],
        dtype=torch.float32,
    )
    start_indices = torch.tensor([[[0, 1]]], dtype=torch.long)

    change_logits, valid = decoder_change_logits(logits, start_indices, ignore_index=-100)

    expected_left = torch.logsumexp(torch.tensor([0.0, -1.0]), dim=0) - 4.0
    expected_right = torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0) - 3.0
    assert torch.allclose(change_logits[0, 0, 0], expected_left)
    assert torch.allclose(change_logits[0, 0, 1], expected_right)
    assert valid.tolist() == [[[True, True]]]


def test_semantic_allocation_loss_prefers_matching_class_and_change_budget() -> None:
    import torch

    from scripts.paper58_benchmark.train_paper58_semantic_auxiliary_latent_probe import (
        semantic_allocation_loss,
    )

    labels = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    start_labels = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
    matching_logits = torch.tensor(
        [
            [
                [[5.0, -5.0], [-5.0, 5.0]],
                [[-5.0, 5.0], [5.0, -5.0]],
            ]
        ],
        dtype=torch.float32,
    )
    mismatched_logits = torch.tensor(
        [
            [
                [[5.0, 5.0], [5.0, 5.0]],
                [[-5.0, -5.0], [-5.0, -5.0]],
            ]
        ],
        dtype=torch.float32,
    )

    matching_loss, matching_parts = semantic_allocation_loss(matching_logits, labels, start_labels)
    mismatched_loss, mismatched_parts = semantic_allocation_loss(mismatched_logits, labels, start_labels)

    assert matching_loss < mismatched_loss
    assert matching_parts["class_quantity_l1"] < mismatched_parts["class_quantity_l1"]
    assert matching_parts["change_quantity_l1"] < mismatched_parts["change_quantity_l1"]


def test_semantic_allocation_loss_ignores_unknown_labels() -> None:
    import torch

    from scripts.paper58_benchmark.train_paper58_semantic_auxiliary_latent_probe import (
        semantic_allocation_loss,
    )

    logits = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    labels = torch.tensor([[[0, -100, 1]]], dtype=torch.long)
    start_labels = torch.tensor([[[0, 0, -100]]], dtype=torch.long)

    loss, parts = semantic_allocation_loss(logits, labels, start_labels, ignore_index=-100)

    assert loss.item() >= 0.0
    assert parts["valid_label_pixels"] == 2
    assert parts["valid_change_pixels"] == 1


def test_filter_cases_excludes_target_neighbor_terms() -> None:
    from scripts.paper58_benchmark.train_paper58_semantic_auxiliary_latent_probe import (
        ExternalSemanticCase,
        filter_cases_by_terms,
    )

    cases = [
        ExternalSemanticCase("baiyangdian_new_area_holdout", None, None, None, None),
        ExternalSemanticCase("sanjiang_plain_holdout", None, None, None, None),
        ExternalSemanticCase("wuxi_taihu_dense_edge_holdout", None, None, None, None),
    ]

    filtered = filter_cases_by_terms(cases, ["baiyangdian", "taihu"])

    assert [case.area for case in filtered] == ["sanjiang_plain_holdout"]
