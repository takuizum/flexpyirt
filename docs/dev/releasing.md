# リリース手順

PyPI への公開は GitHub Actions（`.github/workflows/release.yml`）と
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) で行います。API トークンは使いません。

## 初回だけ行う設定

1. **PyPI に pending publisher を登録する。**
   <https://pypi.org/manage/account/publishing/> で「Add a new pending publisher」:

    | 項目 | 値 |
    |---|---|
    | PyPI Project Name | `flexpyirt` |
    | Owner | `takuizum` |
    | Repository name | `flexpyirt` |
    | Workflow name | `release.yml` |
    | Environment name | `pypi` |

2. **TestPyPI にも同様に登録する**（<https://test.pypi.org/manage/account/publishing/>、Environment name は
   `testpypi`）。リハーサル用で、任意です。
3. **GitHub の Environments を作る。** Settings → Environments で `pypi` と `testpypi` を作成。
   `pypi` には Required reviewers（自分）を付け、公開前に承認を挟むことを推奨します。
4. **GitHub Pages を有効にする。** Settings → Pages → Build and deployment の Source を
   「GitHub Actions」にする。`main` への push で `.github/workflows/docs.yml` がドキュメントを
   <https://takuizum.github.io/flexpyirt/> に公開します。

## 毎回のリリース

1. ROADMAP の当該マイルストーンの受け入れ基準を満たし、`main` の CI がすべて緑であることを確認する。
2. 作業ブランチで以下を更新して PR を出し、`main` にマージする。
    - `pyproject.toml` の `version`（`uv version 0.0.2` など。`uv.lock` も更新される）。
    - `CHANGELOG.md`: `[Unreleased]` の内容を新しいバージョンの節に移し、日付を入れる。末尾のリンクも追加。
    - README / ドキュメントの対応状況（mirt 照合済みの機能だけを載せる）。
3. （任意）Actions → Release → Run workflow で TestPyPI に公開し、確認する:

    ```bash
    uv run --isolated --no-project \
      --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match \
      --with flexpyirt python -c "import flexpyirt; print(flexpyirt.__version__)"
    ```

    TestPyPI は同じバージョンを二度アップロードできないので、リハーサルを繰り返す場合は
    `0.0.1rc1` などのプレリリース番号を使う。

4. `main` の最新コミットにタグを打って push する:

    ```bash
    git switch main && git pull
    git tag -a v0.0.1 -m "v0.0.1"
    git push origin v0.0.1
    ```

5. Release ワークフローが「タグとバージョンの一致確認 → lint / 型 / テスト → ビルド → `twine check` →
   wheel の import 確認 → PyPI 公開 → GitHub Release 作成」を行う。`pypi` 環境に承認者を設定した場合は
   Actions 画面で承認する。
6. <https://pypi.org/project/flexpyirt/> を確認し、別環境で `pip install flexpyirt` して import できることを確かめる。

## 注意

- PyPI に一度上げたバージョン番号は、削除しても再利用できません。間違えたら次のパッチ番号で出し直します。
- sdist にはテスト・mirt 参照値（`tests/reference/`）・R スクリプト（`validation/`）・ドキュメントを含めます。
  wheel には `src/flexpyirt` のみが入ります。
- README は PyPI のプロジェクトページにそのまま表示されるため、リンクは絶対 URL で書きます。
