# 得点の等化（flexpyequate）

[English](equating.md) | 日本語

**flexpyequate** は、同じリポジトリから公開している別のパッケージで、**得点分布**の等化を行う。平均等化、線形等化、
等パーセンタイル等化、アンカーテストを使う方法、対数線形モデルによる事前平滑化、カーネル等化を扱う。依存は NumPy、
SciPy、pandas だけである。flexpyirt をインストールしても入らないので、`pip install flexpyequate` で
インストールする。バージョン番号は flexpyirt と同じである。

項目パラメータを使う等化（IRT 真値等化・観測得点等化、`link()` による尺度のリンキング）は flexpyirt にある。
[ユーザーガイド](guide.ja.md)を参照。

ドキュメントサイトでは、このページの続きにパッケージの
[README](../packages/flexpyequate/README.ja.md)（機能、インストール、使用例、R との一致についての注意）を
表示する。

<!-- include: packages/flexpyequate/README.ja.md -->
