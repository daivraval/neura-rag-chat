import pytest

from neura import pipeline


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_to_groq_preset():
    assert pipeline.resolve_llm() == ("groq", "llama-3.3-70b-versatile")


def test_env_picks_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    assert pipeline.resolve_llm() == ("groq", "llama-3.1-8b-instant")


def test_env_model_does_not_leak_to_another_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    assert pipeline.resolve_llm("huggingface") == ("huggingface", "Qwen/Qwen2.5-72B-Instruct")


def test_explicit_args_win(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    assert pipeline.resolve_llm("groq", "openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")


def test_openai_requires_a_model():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        pipeline.resolve_llm("openai")


def test_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        pipeline.resolve_llm("anthropic-typo")


def test_missing_key_names_the_variable():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        pipeline.make_llm()


def test_client_points_at_the_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = pipeline.make_llm(max_new_tokens=50)
    assert llm.model_name == "llama-3.3-70b-versatile"
    assert llm.openai_api_base == "https://api.groq.com/openai/v1"
    assert llm.temperature == 0 and llm.max_tokens == 50
