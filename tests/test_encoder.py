from types import SimpleNamespace
import numpy as np
import torch
from sallog.encoder import embed_templates, load_embeddings


def test_frozen_encoder_masked_mean_and_l2(config, tmp_path, monkeypatch):
    import transformers
    observations = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["max_length"] == 128
            return {"input_ids": torch.ones(len(texts), 3, dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 0]] * len(texts))}

    class Encoder:
        def eval(self):
            observations.append("eval")
            return self

        def to(self, device):
            return self

        def __call__(self, **batch):
            assert not torch.is_grad_enabled()
            values = torch.tensor([[[1., 0.], [0., 1.], [900., 900.]]]).repeat(len(batch["input_ids"]), 1, 1)
            return SimpleNamespace(last_hidden_state=values)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: Encoder())
    folder = embed_templates(config, tmp_path / "actual_embed")
    mapping, meta = load_embeddings(folder)
    assert observations == ["eval"] and meta["frozen"] is True
    np.testing.assert_allclose(mapping["E1"], [2 ** -.5, 2 ** -.5], atol=1e-6)

