# Contributing

This guide covers everything you need to develop, test, and release changes to rusty-agent-sdk.

## Prerequisites

- **Rust toolchain** (stable) with `rustfmt` and `clippy` components
- **Python 3.9+**
- **uv** (recommended) -- [installation guide](https://docs.astral.sh/uv/)
- **maturin >= 1.9.4** (installed as a dev dependency via uv)

## Dev Setup

```bash
git clone https://github.com/LEO2822/rusty-agent-sdk.git
cd rusty-agent-sdk
uv sync
uv run maturin develop
```

Create a `.env` file in the project root with your API key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

This file is gitignored. The Python examples load it via `python-dotenv`, and the Rust `Provider` reads the key from the environment after dotenv has set it.

## Building

```bash
# Dev build (debug mode, fast compile)
uv run maturin develop

# Release build (optimized, slower compile)
uv run maturin develop --release

# Build a distributable wheel
uv run maturin build --release
```

After any Rust source change, you must re-run `uv run maturin develop` to rebuild the extension module.

## Testing

### Rust Tests

```bash
# Run all unit and integration tests
cargo test

# Run a specific test
cargo test parse_chat_response_returns_first_choice_content
```

### Linting

```bash
# Check formatting
cargo fmt --all -- --check

# Run clippy lints (treat warnings as errors)
cargo clippy --all-targets --all-features -- -D warnings
```

Both checks run in CI and must pass before merging.

## Code Conventions

### Pyclass Layout

- `Provider` and `GenerateResult` live in `provider.rs`
- `TextStream` lives in `stream.rs`
- All three are re-exported from `lib.rs` and registered in the `#[pymodule]`

### Error Mapping

All internal errors use `SdkError` (defined in `errors.rs`):

```rust
SdkError::Connection(msg)  // -> Python ConnectionError
SdkError::Runtime(msg)     // -> Python RuntimeError
SdkError::Value(msg)       // -> Python ValueError
```

Call `.into_pyerr()` to convert an `SdkError` into a `PyErr` at the PyO3 boundary. Keep core logic working with `Result<T, SdkError>`.

### Parameter Pipeline

Python keyword arguments flow through a consistent pipeline:

```
Python kwargs -> build_generation_params() -> GenerationParams -> into_chat_request() -> ChatRequest
```

- `build_generation_params()` in `provider.rs` handles Python type extraction
- `GenerationParams` in `models.rs` is a plain Rust struct (not a pyclass)
- `into_chat_request()` converts to the serializable `ChatRequest`

### Clippy Annotations

Use `#[expect(clippy::too_many_arguments)]` on PyO3 methods that take many parameters. This is intentional -- PyO3 requires flat parameter lists for Python keyword arguments.

## Adding a New Generation Parameter

Follow these steps to add a new parameter (e.g., `top_k`):

### 1. Add field to GenerationParams

In `src/models.rs`, add the field to the `GenerationParams` struct:

```rust
pub struct GenerationParams {
    // ... existing fields ...
    pub top_k: Option<u64>,
}
```

### 2. Add field to ChatRequest

In `src/models.rs`, add the field to `ChatRequest` with the serialization guard:

```rust
pub struct ChatRequest {
    // ... existing fields ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
}
```

### 3. Update into_chat_request()

In `src/models.rs`, pass the new field in `GenerationParams::into_chat_request()`:

```rust
ChatRequest {
    // ... existing fields ...
    top_k: self.top_k,
}
```

### 4. Update build_generation_params()

In `src/provider.rs`, add the parameter to the function signature and struct construction:

```rust
fn build_generation_params(
    // ... existing params ...
    top_k: Option<u64>,
) -> PyResult<GenerationParams> {
    Ok(GenerationParams {
        // ... existing fields ...
        top_k,
    })
}
```

### 5. Update generate_text() and stream_text()

In `src/provider.rs`, add the parameter to both method signatures and their `#[pyo3(signature = (...))]` attributes, then pass it through to `build_generation_params()`.

### 6. Update type stubs

In `rusty_agent_sdk.pyi`, add the parameter to both `generate_text()` and `stream_text()` signatures:

```python
def generate_text(
    self,
    prompt: str | None = None,
    *,
    # ... existing params ...
    top_k: int | None = None,
) -> str: ...
```

### 7. Add tests

In `tests/request_building.rs`, add a test verifying the field serializes correctly and is omitted when `None`.

### 8. Update docs

Update the [API Reference](api-reference.md) with the new parameter.

## Release Process

Version source of truth is `Cargo.toml` (pyproject.toml uses `dynamic = ["version"]`).

```bash
# 1. Bump the version in Cargo.toml
#    e.g., version = "0.2.0"

# 2. Commit the version bump
git add Cargo.toml
git commit -m "bump version to 0.2.0"

# 3. Create a git tag
git tag v0.2.0

# 4. Push the commit and tag
git push origin main --tags
```

CI automatically builds wheels for 8 platforms and publishes to PyPI via OIDC trusted publishing when a `v*` tag is pushed.

## CI Pipeline Overview

The CI workflow (`.github/workflows/CI.yml`) runs on every push and pull request.

### Lint Job

- `cargo fmt --all -- --check` -- formatting
- `cargo clippy --all-targets --all-features -- -D warnings` -- linting
- `cargo test` -- unit and integration tests

### Build Jobs

Builds wheels for all supported platforms:

| Platform | Target | Notes |
|----------|--------|-------|
| Linux x86_64 | manylinux | Standard glibc |
| Linux aarch64 | manylinux | Uses `--zig` for cross-compilation |
| Linux x86_64 | musllinux | Static musl libc |
| Linux aarch64 | musllinux | Uses `--zig` for cross-compilation |
| Windows | x64 | MSVC target |
| macOS | x86_64 | Intel |
| macOS | aarch64 | Apple Silicon |
| sdist | source | Fallback source distribution |

### Smoke Test

After building, installs the wheel in a clean environment and verifies:

- `import rusty_agent_sdk` succeeds
- `Provider` can be instantiated (with a dummy API key)

### Release

Only runs on `v*` tags. Publishes all wheels and the sdist to PyPI using OIDC trusted publishing (no API tokens stored in secrets).

## Stale Cache Workaround

If you see stale behavior after Rust source changes (old behavior persists despite rebuilding), uv may be serving a cached `.so` file:

```bash
uv cache clean rusty-agent-sdk
uv run maturin develop
```

## Related Documentation

- [Architecture](architecture.md)
- [API Reference](api-reference.md)
- [Getting Started](getting-started.md)
