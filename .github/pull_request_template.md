## 概要
<!-- 何を・なぜ変えたか -->

## 種類
- [ ] feature（`feature/<slug>` → `develop`）
- [ ] fix（`fix/<issue番号>-<slug>` → `develop`）— 下に `Closes #<issue番号>` を書く
- [ ] release（`develop` → `main`）— タイトルを `release(major|minor|patch): ...` にする

Closes #

## 確認
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` が通る
- [ ] 数値計算の変更なら mirt 照合テストを追加・更新した（docs/VALIDATION.md）
- [ ] 公開 API を変えたなら docs/API.md と README を更新した
