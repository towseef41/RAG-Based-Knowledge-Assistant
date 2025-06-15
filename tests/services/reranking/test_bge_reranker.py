import pytest
from unittest.mock import patch, MagicMock
import torch
from app.services.reranking.bge_raranker import BgeReranker

@patch("app.services.reranking.bge_raranker.AutoTokenizer.from_pretrained")
@patch("app.services.reranking.bge_raranker.AutoModelForSequenceClassification.from_pretrained")
def test_bge_reranker_rerank(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    # ✅ Step 1: Mock the tokenizer and make it callable
    mock_tokenizer = MagicMock()
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]])
    }
    mock_tokenizer.side_effect = lambda pairs, **kwargs: mock_tokenizer.return_value

    # ✅ Step 2: Mock the model and its return value
    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.9], [0.3]])
    mock_model.side_effect = lambda **kwargs: mock_output

    # ✅ Step 3: Use the reranker
    reranker = BgeReranker(model_name="dummy-model")
    docs = [{"text": "doc one"}, {"text": "doc two"}]
    result = reranker.rerank("test query", docs)

    # ✅ Step 4: Assertions
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["score"] >= result[1]["score"]
    assert all("text" in r and "score" in r for r in result)
