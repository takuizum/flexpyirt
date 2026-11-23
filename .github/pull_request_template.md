## Summary / 概要
<!-- What changed and why / 何を・なぜ変えたか -->

## Type / 種類
- [ ] feature (`feature/<slug>` → `develop`)
- [ ] fix (`fix/<issue number>-<slug>` → `develop`) — add `Closes #<issue number>` below / 下に `Closes #<issue番号>` を書く
- [ ] release (`develop` → `main`) — title `release(major|minor|patch): ...` / タイトルを `release(major|minor|patch): ...` にする

Closes #

## Checks / 確認
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` passes / が通る
- [ ] Numerical changes come with mirt agreement tests (docs/VALIDATION.md) / 数値計算の変更なら mirt 照合テストを追加・更新した
- [ ] Public API changes are reflected in docs/reference and README, in English and Japanese / 公開 API を変えたなら docs/reference と README を英日で更新した
