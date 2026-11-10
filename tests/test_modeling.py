import torch
import torch.nn.functional as F
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from rte.modeling import factorize_embeddings, load_checkpoint
from rte.params import count_params
from rte.train import kd_loss


def tiny_deberta():
    torch.manual_seed(0)
    conf = DebertaV2Config(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
        pad_token_id=0,
    )
    return DebertaV2ForSequenceClassification(conf).eval()


def test_kd_loss_matches_hinton_formula_and_masks_teacher_errors():
    s, t, y = torch.randn(6, 2), torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 1, 0])
    ref = 0.5 * F.cross_entropy(s, y) + 0.5 * 4 * F.kl_div(
        F.log_softmax(s / 2, -1), F.softmax(t / 2, -1), reduction="batchmean"
    )
    assert torch.allclose(kd_loss(s, t, y, 0.5, 2.0), ref, atol=1e-6)
    wrong_teacher = torch.stack([y.float(), 1 - y.float()], -1) * 10  # argmax != y everywhere
    masked = kd_loss(s, wrong_teacher, y, 0.5, 2.0, mask_teacher_errors=True)
    assert torch.allclose(masked, F.cross_entropy(s, y))


def test_full_rank_factorisation_is_exact_and_counts_params():
    model = tiny_deberta()
    ids = torch.randint(1, 50, (3, 7))
    before = model(input_ids=ids).logits
    kept = factorize_embeddings(model, rank=16)  # full rank: hidden size
    assert abs(kept - 1.0) < 1e-5
    assert torch.allclose(model(input_ids=ids).logits, before, atol=1e-4)
    assert count_params(model)["word_embedding"] == 50 * 16 + 16 * 16


def test_factorised_checkpoint_round_trip(tmp_path):
    model = tiny_deberta()
    factorize_embeddings(model, rank=4)
    model.save_pretrained(tmp_path)
    ids = torch.randint(1, 50, (2, 5))
    loaded = load_checkpoint(tmp_path).eval()
    assert loaded.config.embedding_rank == 4
    assert torch.allclose(loaded(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-6)
