# CrystOD 開発記録 (Development.md)

AI(Claude)との協働開発の記録・引き継ぎ資料。対象期間: **2026-07-07 〜 07-11**、バージョン: **v0.2.1 → v0.3.1**。
「どんなエラーが起きたか / どう対処したか / どんな要望があったか / 今後何ができるとなお良いか」をまとめる。
今後の開発(人間・AI どちらでも)は、この文書と README.md の Changelog、testsuite.py を読めば背景と検証方法を再現できるはず。

## 開発体制と環境

- **分担**: 小磯くん(Institute of Science Tokyo)が PLAN.md で仕様を書き、望月が AI コーディングで実装。`script/` 以下の個別 script をパッケージへ移植していく方式(移植後は**パッケージ側を正**とし、元 script は未修整で残置)。
- **環境**: conda env `crystod`(`~/anaconda3/envs/crystod`, Python 3.11)。editable install なので `~/CrystOD-main` のソース修正が即反映される。エントリポイント更新は `~/anaconda3/envs/crystod/bin/pip install -e . --no-deps`。
  **注意**: Mac 標準の python3.9(`/usr/local`)にも同リポジトリの古い editable install が残っており、取り違えると挙動が変わる。テスト・動作確認は必ず conda env で行うこと。
- **テスト**: `testsuite.py`(全 27 セクション・310 チェック)。修正のたびにフルスイートで回帰確認。セクション番号指定で部分実行可(`python testsuite.py 26`)。
- **リポジトリは現状 git 管理外**(→ §5)。

## セッション一覧

| 日付 | テーマ | 主な成果 |
|---|---|---|
| 7/7–7/8 | フォノン状態可視化ほか機能追加 | v0.2.2–v0.2.3: `--phonon-vector`/`--fatband`/`--lt`/`--bz(-supercell)`/`--xdatcar2adp`/`--decompose-irrep`/`--ligand-field-split`/`--spin-basis`(SAMM) |
| 7/9–7/10 | CLI 全面再構成(usability) | v0.3.0: phonopy 風セクション化(`crystod` + `crystod-bz/md/mag/phonon/group`)、SALC HTML ビューワー刷新、モード番号 1 始まり統一、旧フラット形式の完全削除 |
| 7/10–7/11 | modulation / irrep ラベリング改修 | v0.3.1: modulation 振動数バグ修正、CDML ラベル表示、star arm 対応、分数座標対応 |
| 7/11 | f 軌道資料の作成 | 32 点群×f 軌道の既約表現分解と基底関数の一覧(Word/PDF)。点群「1」クラッシュ修正 |
| 7/22 | 非特殊 k 点の irrep ラベル(ISO-IR fallback) | `crystod/isoir.py` 新設: Stokes–Campbell ISO-IR (CIR/PIR) テーブルのパーサ+ラベラー。irreptables に無い k(線・面・一般点)で Miller–Love ラベル(T1, DT5, GP1 等)を出力(→ §3, §5-15) |
| 7/22 (2) | ISO-IR fallback の全経路展開 | `crystod --atomic-orbital`・`crystod-mag`・`crystod-phonon --vibration/--vector/--modulation` にも展開。共有ヘルパー(`isoir.get_isoir_label_map` 等)+ phonopy バンド組(可約指標)用の分解型照合 `decompose_characters` を追加。q 点名も ISO k 型で表示(旧 "custom"/"-") |
| 7/24 | データのパッケージ内移動 + q 点名のファイル名反映 | `CIR_data.txt.gz` を `crystod/` 直下へ移動(pyproject package-data 登録、探索順は env → パッケージ → `ISOTROPY/`)。`--vector`/`--modulation` の非特殊 q のファイル名・表示を ISO k 型に(例: `POSCAR_MoS2_U_mode1_U2_conv.vesta`、`MPOSCAR_DT_mode1_DT3_Cmcm`。旧 `q_0.5_0_0.2` 形式は ISO 判定不能時のみ)。同一線上の複数 q の上書き回避用に `--keep-q-coords` を追加。README(Features/§2/§24/§25/§26/§28/Changelog)整備、**v0.3.4** |
| 7/24 (2) | `--irreps` の k-path 中点 | `phonon_irreps.yaml` に seekpath k-path 各セグメントの中点(対称線 DT/Z/SM/LD/S/T 等)の irrep を ISO-IR ラベルで追加(`k_path:`・`path_midpoints:` ヘッダ + `segment:` 付き irrep ブロック)。Pm-3m(ScF3/SrTiO3)・Fd-3m(Si、F 格子)・P4mm(BaTiO3、9 セグメント)で検証。中点は **`--all-irreps` オプトイン**(デフォルトは従来の特殊点のみ = 高速 1.6 秒; --all-irreps は family 探索のバッチ化 `decompose_characters_many` で 36 → 8 秒)。出力先も分離: デフォルト `phonon_irreps.yaml` / `--all-irreps` `phonon_irreps_all.yaml`(両者が同一 directory に共存できる)。実行末尾に出力ファイル名を print |
| 7/27 | `--basis`/`--generate-basis` の ISO-IR fallback + R 底心バグ修正 | `basis_function._analyze_space_group` に ISO-IR fallback を接続(構造ファイルが無いため、irreptables の対称操作から**合成 conventional セル**(汎用格子+一般点2軌道)を構成して `IsoIRLabeler` に渡す方式。BCS↔ISO の setting 差も spglib 標準化が自動吸収)。例: `--basis x y z --sg Pm-3m --kpoint 0 0 0.1` → `DT [0,0,0.1]`, `DT1(1)+DT5(2)`。P/C/F/I/R 底心・非共形・2 origin 群で検証。**既存バグ発見・修正**: conventional→primitive の並進変換が row 規約(`t @ Minv`)で回転の column 規約と混在し、**非対称な R 底心行列で群の閉包が破れていた**(R-3c で 48/144 積が違反 → spgrep への入力群が壊れ、R 系空間群の --basis 特殊点ラベルも壊れていた)。pure column(`Minv @ t`)に統一して修正(§1.3)。ISO fallback 成功時の誤解を招く UserWarning も抑制。残件 2 件をタスク化: 非共形 little group での Bloch 位相欠落(多重度が厳密整数でない; 境界特殊点で空/部分出力)、--product --sg 線分項の ISO 命名 |
| 7/27 (2) | `--table` の空間群対応 | `crystod-group --table --space-group SG --kpoint ...` で k の little group の指標表を表示(点群 `--table` の空間群版)。`_analyze_space_group` の前半を `_spacegroup_irrep_context` に共通化し、`format_spacegroup_table` を新設。表引き k は BCS ラベル、線・面は ISO-IR ラベル、列見出しは Seitz 記号。Pm-3m GM/T 線・Pnma X(複素型)・番号指定(--sg 221)で検証 |
| 7/27 (3) | symmetry mode の irrep 別 VESTA 可視化 + `--atomic-orbital` のハイフン対応 | `--supergroup-cif` 実行時に、活性 irrep ごとの変位パターンを `{supergroup_file}_{irrep}.vesta`(親由来参照構造 + 矢印、invariant-core セル、最大矢 1.5 Å)として自動出力(`symmetry_mode._export_mode_vesta_files`、`phonon_vector.write_vesta_with_arrows` を再利用)。CaTiO3 Pnma の 5 モード(X5+/M2+/M3+/R4+/R5+)で元素選択則を検証(R4+/M3+/M2+ = O のみ、X5+/R5+ = Ca 主体、Ti は全モード不動)。example に CaTiO3 の .vesta 一式を同梱。`--atomic-orbital` は `Ti-d`(ハイフン)も受理(出力はバイト同一、エコーは正規形 `Ti_d`)。testsuite 16 は全 run を tempdir 実行に変更(repo 直下への .vesta 混入防止) |
| 7/27 (4) | symmetry mode VESTA の `--conventional` | `--supergroup-cif --conventional` で mode VESTA を親 conventional 基底で出力(`_conv` サフィックス、`--vector` と同一セマンティクス)。表示セルは「全 mode パターンを周期的に収める conventional 形状の最小対角倍セル」(D 行が invariant-core 格子の元、`_conventional_display_cell`)。La3Ni2O7 I4/mmm→Cmcm で検証: `139_..._X3-_conv.vesta` = 90° conventional metric (c=20.07 Å、面内は X3- の反位相を収めるため 2 倍)、O 主体 + La/Ni 小 = 成分表と一致、GM1+ は中心 La 不動 |

---

## 1. 発生したエラー・バグと対処法

### 1.1 環境・依存ライブラリ起因

| 症状 | 原因 | 対処 |
|---|---|---|
| `AttributeError: 'dict' object has no attribute 'international'`(複数箇所で再発) | spglib の symmetry dataset がバージョンにより dict / オブジェクトで返る | アクセスをバージョン非依存のアクセサ経由に統一(互換ヘルパーに集約) |
| `AttributeError: 'IrReps' object has no attribute 'frequencies'` | phonopy ≥ 2.21 で同プロパティが削除 | 新 API に対応 |
| `AttributeError: module 'pymatgen' has no attribute '__version__'`(xdatcar2adp 移植時) | 移植元 script の pymatgen 依存と環境差 | XDATCAR パーサを自前実装し pymatgen 依存を除去 |
| reportlab の `TTFError`(ヒラギノは PostScript アウトラインで埋め込み不可) | 日本語フォント埋め込みの制約 | reportlab を使わない PDF 生成に切替(7/11 の資料は docx 生成 → Microsoft Word の AppleScript 変換。Word はサンドボックスのため出力先は `~/Library/Containers/com.microsoft.Word/Data/` 配下を使う) |
| コマンド・テストが Mac 標準 python3.9 で実行されてしまう | 2 つの editable install が共存 | 「conda `crystod` 環境で作成・検証」を運用ルール化 |

**教訓**: 環境依存の API 差(spglib/phonopy)は互換ヘルパーに集約する。外部依存はなるべく増やさない方針(ASE への置き換え、pymatgen 除去。PLAN.md 参照)。

### 1.2 CLI・argparse 起因

| 症状 | 原因 | 対処 |
|---|---|---|
| `--point-group -43m` が `expected one argument` で拒否 | `-` 始まりの値を argparse がオプションと誤認 | CLI 前処理で `--point-group`/`--subgroup` 直後の `-` 始まりトークンを `=` 形式に自動結合(`-43m`/`-42m`/`-6m2`/`-1` 対応、testsuite に検証追加) |
| `--dim` が 3 値/9 値、引用符あり/なしで挙動が揺れる | パーサの形式想定が狭い | `"4 4 4"`・`4 4 4`・9 値対角行列すべて受理。非対角は明示的エラー |
| `--kpoint M`(ラベル指定)が文書化されているのにパース失敗 | 旧トップパーサの `nargs=3` 固定 | v0.3.0 再設計で `nargs='+'` 化、GM/X/M/R ラベル受理 |
| セクション化作業中の `ModuleNotFoundError`(`crystod.cli.brillouin_zone` 等) | モジュール分割・リネーム時の参照不整合 | `cli.py` → `crystod/cli/` パッケージ化の過程で import を整理 |

### 1.3 表現論・数値計算起因

#### `--modulation` のモード表の振動数が間違っていた(v0.3.1、最重要バグ)

- **症状**: `crystod-phonon --modulation --qpoint 0 0 0.5`(Sr3Ti2O7, I4/mmm)のモード表の振動数が、`--irreps` の phonon_irreps.yaml(= phonopy の直接対角化)と一致しない。一部のモードだけは一致していた。
- **原因**: 旧 `modulation.py` は既約表現に射影した各ブロックを**個別に**対角化しており、同じ既約表現が複数回現れる q 点での**ブロック間結合を無視**していた。一致していたのは「その q 点で 1 回しか現れない既約表現」(X1-/X2+ など)だけ(対角ブロックは Schur の補題によりスカラーなので、その場合のみ正しい)。
- **対処**: `phonon_vector.py` の `build_symmetry_adapted_modes` が既に持っていた正しい構成(等価既約表現ブロックのクラスタリング → intertwiner で基底整列 → multiplicity 次元の結合行列を対角化)を `SymmetryAdaptedModulation.__init__` に移植。さらに**毎回の実行時に phonopy スペクトルとの一致(atol=1e-3 THz)とモードベクトルが真の固有ベクトルであることを内部検証**し、ズレたら RuntimeError で落とす。
- **教訓**: 「一部だけ一致する」パターンは縮退・多重度がらみの取りこぼしを疑う。同じ物理を二箇所で別実装していたのが根本原因(共通化して解消)。
- **注意**: この修正により、該当 q 点では mode 番号↔irrep の対応が v0.3.0 以前と変わっている。過去に作った MPOSCAR を使い回すときは要確認。

#### irreptables が star の代表 arm しか照合できない(v0.3.1)

- **症状**: Pm-3m で (0, 1/2, 0) だけが X と判定され (1/2, 0, 0)・(0, 0, 1/2) は判定されない。I4/mmm primitive の X の 2 arm も同様。SALC では分解が汎用名 `irrep_N` に落ち、`--star-of-k` では k 点名が `custom` になる。
- **原因**: irreptables/seekpath は各特殊点につき**代表 arm の座標 1 つ**しか表引きできず、コード側が完全一致でしか照合していなかった。
- **対処**(共通ヘルパーを `crystod/operations.py` に集約):
  - `find_star_arm(k, rotations, special_points)` — 空間群回転 g で `k' = k g`(逆格子並進を法とする)が表引き可能な代表点に到達する g を探索。
  - **phonon 系**(`--modulation`/`--vector`): q を代表点に写してから表引き。star の各 arm のスペクトルはバンドごとに同一なので、代表点のラベルをバンド番号対応で適用。
  - **SALC/軌道系**: little group の指標を演算ごとに照合するため、共役同型 `h → g⁻¹hg`(G_k → G_k_rep)で演算を輸送する `conjugated_little_group_map` を実装。Seitz 積で生じる格子並進差 Δ の Bloch 位相 `e^{-2πi k_rep·Δ}` を指標に掛けて補正。
  - 非共形群での検証: Fd-3m Si の X star(並進位相 ±1)と W star(|star|=6、複素位相域)で全 arm が代表点と同一ラベルになることを確認。
  - **ハマりどころ**: `phonon.symmetry` は**スーパーセル**の対称性(4×4×4 なら 3072 個)。primitive 基底の回転は `phonon.primitive_symmetry` から取ること。
- **適用範囲**: SALC 本体 / `--atomic-orbital` / `--visualize` / `--star-of-k` / `crystod-mag` / `--vibration` / `--vector` / `--modulation` の全ラベル引き経路。

#### 1/3 座標が 0.333333 に丸められて認識されない(v0.3.1、昔からの問題)

- **症状**: `crystod-mag -c 185_PPOSCAR_LuFeO3 --element Fe`(P6_3cm)で K (1/3,1/3,0)・H (1/3,1/3,1/2) だけ汎用ラベル + 誤った分解になる。また `--qpoint 1/3 1/3 0` は `ValueError: could not convert string to float: '1/3'`。
- **原因**: 特殊点リスト生成が表示用の 6 桁丸め(0.333333)を計算にそのまま流していた。1/3 から 3.3×10⁻⁷ のずれで spgrep の little group 検出が位数 6 → 2 に縮退。
- **対処**(`crystod/operations.py`): `snap_qpoint`(表由来座標を分母 ≤ 48 の単純分数へ無条件スナップ)と `parse_qpoint_token`(ユーザー入力の `1/3` 分数表記を受理、1×10⁻⁶ 以内の小数のみスナップ — 0.34 のような一般点は変えない)。なお `crystal_orbital_spgrep.canonicalize_kpoint` は最初から分数スナップ方式で正しく、方式をそちらに揃えた形。

#### 2 次元既約表現の 2 成分が同一の VESTA ファイルになる(v0.3.1)

- **症状**: La3Ni2O7 (I4/mmm) の `crystod-mag` で `..._GM5+_dipole_x_conv.vesta` と `..._x_conv_2.vesta` が完全に同一(直交する x/y 成分になるはず)。
- **原因**: spgrep の射影が GM5± を円偏光型の複素基底 (x±iy) で返し、旧 `_realify`(行ごとの位相回転のみ)が失敗 → `np.real()` が虚部(y 情報)を捨てて両成分が ±x に潰れた。
- **対処**: `spin_basis._realify` を拡張 — 位相回転で実化できない場合、実部・虚部のグラム・シュミットで実直交基底を再構成(空間が複素共役で閉じている場合に可能)し、張る空間の同一性を射影検証。`dipole_x`/`dipole_y` の直交対が出力されるように。

#### `--basis` の conventional→primitive 並進変換が R 底心で群を壊していた(7/27)

- **症状**: `crystod-group --basis x y z --space-group R-3c` 系(R 底心)で、特殊点・線上とも irrep ラベルが汎用名に落ちる/照合失敗。
- **原因**: `basis_function._analyze_space_group` の primitive 変換が、回転は column 規約(`Minv @ R @ M`)、並進は row 規約(`t @ Minv`)と**混在**。両者は対称な底心行列(P/F/I)では一致するが、**非対称な R 行列では不一致**で、生成した primitive 操作系が群として閉じない(R-3c で 144 積中 48 が閉包違反)→ spgrep への入力自体が壊れていた。C も非対称だが並進の値の巡り合わせで偶然閉じていた。
- **対処**: 並進も pure column(`Minv @ t`)に統一(閉包違反 0 を確認)。k の変換 `k @ Minv` は column 規約の双対(行ベクトル)なのでそのままで正しい。
- **教訓**: `spacegroup_product.SpaceGroupIrrepAlgebra` は同じ変換を**実行時に群閉包で自己検証**しており無事だった。規約が交錯する変換は閉包チェックを実装側に持たせるのが安全。
- **関連する未解決の既存問題**(タスク化済み): `--basis` の可約表現指標は Bloch 並進位相を含まないため、非共形 little group では多重度が厳密整数にならない(内点では丸めで隠れ、非共形群のゾーン境界特殊点 — Pnma X, Fd-3m X, R-3c T — では空/部分的な分解が出る)。

#### その他(v0.2.x〜7/11)

| 症状 | 原因 | 対処 |
|---|---|---|
| `--decompose-irrep` で `TypeError: only 0-dimensional arrays ...` | script 移植時の `np.dot` 次元不整合 | 配列次元処理を修正 |
| Wigner-D 関連の複数バグ(v0.2.1) | improper 操作のパリティ因子(det vs detˡ)、β=π の Euler 角分岐、p_y 行の符号、行列式の float 摂動 | `operations.py` を純 NumPy に書き直して一括修正 |
| 空間群 basis-function の係数が数値ノイズで発散(v0.2.1) | 射影の数値誤差 | ノイズ除去を実装 |
| 点群「1」で `crystod-group --basis` がクラッシュ(`IndexError: 0-dimensional array`、7/11) | 自明群では指標表の値がスカラーになる | `basis_function.py` の `_project_irrep_basis` 冒頭で `np.atleast_1d` により正規化 |

### 1.4 可視化起因

| 症状 | 原因 | 対処 |
|---|---|---|
| `--phonon-vector` の変位矢印が VESTA に出ない/斜めに傾く | 固有ベクトルの位相・縮退モードの任意結合 | `--modulation` と同じ対称性適応固有ベクトル方式で実装し直し(縮退モードは対称性が定める方向に整列) |
| **plotly ドラッグ回転が初期位置にスナップバック**(要注意の落とし穴) | `plotly_relayouting`(ドラッグ中)内で `Plotly.relayout()` を呼ぶと、古いカメラでメインシーンが再描画されドラッグがキャンセルされる | ドラッグ中はコンパス側シーンを内部 API `scene._scene.setViewport()` のみで同期し、終了時に `layout.scene2.camera` へ書き戻す |
| 半透明ローブの前後関係が破綻(奥が手前に見える) | WebGL/plotly は半透明サーフェス同士の depth sort ができない(原理的制約) | デフォルト不透明(opacity 1.0)+ VESTA 風ライティング。スライダーで半透明にした時のみ視線方向射影の depth fade で奥行きを擬似表現 |
| SALC HTML が 20MB/78MB と肥大 | 波動関数の値をグリッド全点で埋め込み | 符号のみ 2 色(+黄 #ffeb3b / −シアン #26c6da、VESTA 準拠)、座標 3 桁丸め、グリッド 22×44 → 約 1/5(78M→16M) |

### 1.5 運用上の教訓

- **testsuite の一過性失敗**: バックグラウンド実行 1 回だけ「274 passed, 30 failed」が出たが、出力を `tail -3` にパイプしていたため失敗内容が失われた(直後 2 回連続全合格で一過性と判断)。**バックグラウンドのテスト実行はパイプせず全出力をファイルに残すこと。**
- **example/README の陳腐化**: 修正で数値・ファイル名が変わるたびに example 内の参照出力を再生成して差し替える必要がある。

---

## 2. ユーザー要望の履歴(何を求めてきたか)

### v0.2.2–v0.2.3(7/7–7/8): script のパッケージ統合

1. フォノン固有ベクトルの VESTA 可視化(`--phonon-vector`、旧称 `--phonon-mode` から改名)。複数モード合算、`--conventional`
2. 既約表現分解の対話ツール(`--decompose-irrep`、小磯くん script)
3. 配位子場分裂(`--ligand-field-split`、小磯くん script)
4. XDATCAR→ADP CIF(`--xdatcar2adp`、佐藤さん script)
5. 元素射影 fatband(`--phonon-fatband`): VESTA 元素色、図サイズ・フォント調整、`--nac`(LO/TO)
6. BZ 3D プロット(`--bz`/`--bz-supercell`、小磯くん script)
7. L/T 分別フォノンバンド(`--phonon-lt`)
8. **スピン多極子基底(SAMM、鈴木通人先生の論文準拠)**: `--spin-basis`。AlNi3 の Ni 反強磁性配列探索から出発し、irrep ラベル・MAGMOM 出力・VESTA 出力へ発展。軸性ベクトル基底(Rx, Ry, Rz)の `--basis-function` 対応もここから

### v0.3.0(7/9–7/10): CLI 再構成 — 2026-07-08 ミーティング決定

- フラットな `--<mode>` フラグ約 18 種が `--help` を読めなくしていた → **phonopy 流のセクションコマンド**へ: `crystod`(メイン = 売りの SALC)+ `crystod-bz` / `crystod-md` / `crystod-mag` / `crystod-phonon` / `crystod-group`
- オプション名を phonopy 準拠に(`-c/--cell`、`--dim`、`--nac`、`--band-labels`)
- 一旦 deprecation シムを挟んだ後、**旧フラット形式は完全削除**(望月の判断。旧フラグは等価コマンドを案内するエラーに)
- **モード番号を全コマンドで 1 始まりに統一**(表示・ファイル名・入力すべて。内部は 0 始まりのまま CLI 境界で変換)
- SALC HTML ビューワーを **Henrique Miranda の phononwebsite 風**にリデザイン(BSD-3-Clause、模倣を README に明記)。VESTA 風の結合・配位多面体(`--bond EL1 EL2 MAX`)、境界原子の周期像補完、a/b/c コンパス、軽量化
- 出力ファイルの自動命名規則整備(§4)
- `crystod-mag` は `--show-spin-direction` なしでもデフォルトで MAGMOM 出力、`--format vasp|qe`

### v0.3.1(7/10–7/11): modulation / ラベリング

| # | 要望 | 実装 |
|---|------|------|
| 1 | `--modulation` で `--mode` を決める前にモード表と Star of q を見たい | `--qpoint` のみで実行するとモード表 + Star of q を表示して終了する「プレビューモード」 |
| 2 | 表の `Irrep Block`(内部番号)を実ラベルにしたい | CDML ラベル(X3-(1), GM5-(2), ...)を irreptables ベースで表示。取れない q 点は `-` |
| 3 | 振動数が `--irreps` と合わない | → §1.3 のバグ発見・修正 |
| 4 | 出力名を `MPOSCAR_{qpoint}_{mode}_{irrep}_{subgroup}` に | 例: `MPOSCAR_X_mode1_X3-_Cmcm`。マルチ q は項ごとに連結 |
| 5 | star のどの arm でもラベルを引きたい(SALC 含む) | → §1.3 |
| 6 | `--star-of-k` のヘッダーにもラベルを | `custom` → `X [0.0, 0.5, 0.5]` |
| 7 | 1/3 座標の認識失敗を解決 | → §1.3 |
| 8 | 2 次元既約表現の 2 成分は直交するはず | → §1.3 |
| 9 | README の Command Summary / Changelog 更新 | 都度更新 + バージョン 0.3.0 → 0.3.1 |

### 7/11: f 軌道資料

- **32 点群すべてにおける f 軌道基底関数の既約表現分解**の Word/PDF 資料(指標表・書籍に載っていないため)。`crystod-group --basis` を全点群に適用。一般セット(tesseral)と立方晶用 cubic セットの使い分け(一般セットを立方晶に与えると r²·p が混入した 10 次元空間に閉包する)、両セットのユニタリ変換関係 f_{x³} = −(√6/4)f_{xz²} + (√10/4)f_{x(x²−3y²)} を掲載。

---

## 3. 主要な技術要素(今後の参照用)

- **等価既約表現クラスタの対角化**(`modulation.py`, `phonon_vector.py`): 射影ブロック間の結合を検出してクラスタ化し、intertwiner で基底を整列すると結合行列がスカラー×単位行列(Schur)になり、multiplicity 次元の固有値問題に落ちる。実行時に phonopy スペクトルと照合する自己検証つき。
- **star arm の共役輸送**(`operations.find_star_arm`, `operations.conjugated_little_group_map`): `k g = k_rep` なる g に対し `h → g⁻¹hg` は G_k → G_k_rep の同型。指標は χ_k(h) = χ_rep(g⁻¹hg) × e^{−2πi k_rep·Δ}(Δ は Seitz 積と表の並進の差 = 格子並進)。
- **分数スナップ**(`operations.snap_qpoint`, `operations.parse_qpoint_token`): 表由来座標は無条件スナップ(分母 ≤ 48)、ユーザー入力は許容誤差 1e-6 内のみスナップ。
- **複素 2 次元空間の実化**(`spin_basis._realify`): 位相回転 → 失敗したら Re/Im のグラム・シュミット再結合 → 張る空間の同一性を射影検証。
- **rotations の基底に注意**: `phonon.symmetry` はスーパーセルの対称性。primitive 基底の回転は `phonon.primitive_symmetry` から取る。
- **f 軌道の基底セット**: 非立方晶 27 点群は tesseral セット、立方晶 5 点群は cubic セット(`--basis` に与える関数系が異なる)。詳細は 7/11 作成の資料参照。
- **ISO-IR (ISOTROPY) テーブルによる非特殊 k のラベリング**(`crystod/isoir.py`, 2026-07-22):
  - データ: **パッケージ内 `crystod/CIR_data.txt.gz`**(複素既約表現、全 k 型: 点・線・面・一般点。gzip で 53MB → 1.1MB、pyproject の package-data に登録済み → 2026-07-24 移動)を遅延パース。探索順は `CRYSTOD_ISOIR_PATH` 環境変数 → パッケージディレクトリ → リポジトリ直下 `ISOTROPY/`(ISO-IR 配布レイアウト `CIR_data/CIR_data.txt[.gz]`、後方互換)。SSG/PIR/xlsx を含むフルセットは `~/CrystOD-main_trial/ISOTROPY/` に保管。PIR(physically irreducible)も同じパーサで読める(k ベクトル数が CIR=kcount、PIR=pmkcount の点だけ異なる)。
  - ISO-IR は各既約表現の **full 表現行列**(star 全 arm、標準 conventional setting の 48 個以下の代表操作)を持つ。arm j の対角ブロック × 並進位相 e^{+2πi k·t} が小表現。非特殊 k は α,β,γ パラメータ付きで指標を評価できる。
  - **位相規約の橋渡し**: spgrep/irreptables は e^{−2πi k·t}、ISO-IR は e^{+2πi k·t}。よって spgrep 指標は **ISO-IR 指標の複素共役**と照合する。複素型既約表現ではこの規約差が**ラベルの入れ替わり**として現れる(例: Pnma R 点は Bilbao R1 = ISOTROPY R2)。実指標の k 点では両規約のラベルは一致(Pm-3m の R/M/X で確認済み)。
  - **setting の整合**: ISOTROPY の標準 setting(origin choice 2、直方晶 abc、単斜 b 軸 cell choice 1、六方軸)へ spglib の `hall_number` 指定(選択則: choice '2' → '' → 'b1' → 'H' → '1')で決定論的に変換。origin choice の異なる群(Fd-3m 等)でも正しく動く。※mod 格子のグリッド探索による origin 推定は擬シフトを拾い誤ラベルの危険があるため不採用(68 Ccce の T 点で実証)。
  - **検証**: (i) T 線 C4v 指標の直交性、(ii) ISO-IR 行列のみでのバンド表現の独立分解が crystod 出力と一致、(iii) ISOSUBGROUP 由来の実証的対応表(141 X, 142 X, 230 N)と一致、(iv) SG 68 T ではゲージ非依存の isotropy subgroup 構成(spgrep 行列 → 子群同定)で直接検証。
  - 適用先: 7/22 (2) 以降は全ラベリング経路(`crystod --element/--orbital`・`--atomic-orbital`・`crystod-mag`・`crystod-phonon --vibration/--vector/--modulation`)。irreptables に載っている k 点(star-arm 輸送含む)では従来通り irreptables が優先(出力不変)。spinor は ISO-IR に二重群が無いためスキップ(従来通り汎用ラベル)。

---

## 4. 開発規約(このリポジトリの慣例)

- **新機能は対応するセクションコマンドに追加**する(新しいトップレベル `--<mode>` フラグは作らない)。手順: `crystod/cli/` にモジュール → `pyproject.toml [project.scripts]` → testsuite に番号付きセクション追加 → README 更新(Features / 該当セクション / Command Summary / Changelog + バージョンアップ)→ example の参照出力が変わる場合は再生成。
- **モード番号は 1 始まり**(全コマンド・全出力。今後のモード選択系機能も必ず)。
- **ファイル命名規則**:
  - フォノン: `POSCAR_<formula>_<qlabel>_mode<NN>_<irrep>[_conv].vesta`(番号ゼロ埋めで ls 昇順、irrep の次元サフィックス除去、連結は `+`)
  - モード表: `phonon_modes_<formula>_<qlabel>.txt` 自動保存
  - SALC: `SALC_{element}_{orbital}_{kpoint}.html` 自動出力
  - modulation: `MPOSCAR_{q}_{mode}_{irrep}_{subgroup}`
- **`--conventional`** は crystod --visualize / crystod-phonon --vector / crystod-mag で同一セマンティクス(centring 由来の primitive→conventional 行列、`_conv` サフィックス)。
- **ビューワーの文言**: トップバー「Symmetry-Adapted Linear Combination (SALC) viewer」、サイドバー見出し「Irreps of SALC」。
- **検証の作法**: 対称性がらみの修正は、共形群(Pm-3m)と非共形群(Fd-3m)、1 次元と多次元既約表現、代表 arm と非代表 arm、のように**壊れ方が異なる軸で複数ケース**を確認する。可視化系はブラウザで実際にドラッグ・表示確認。
- **バージョン**: `pyproject.toml` と `crystod/__init__.py` の 2 箇所(`crystod --version` で確認)。

---

## 5. これができるとなお良し(今後の改善候補)

### 決定待ち・体制
1. **群論電卓コマンドの名称確定**: 暫定 `crystod-group`。小磯くんは `crystod-rep` 推し、望月は本体 `crystod` への統合も検討中 → 要確定(セクション分担は 2026 年 7 月末締切)。
2. **リポジトリの git 管理化**: 現在 git 管理外。7/11 の点群「1」修正も履歴に残っていない。`git init` して現状を初期コミットにすることを強く推奨。
3. **元 script との乖離**: `script/` のオリジナルは移植後も未修整で残置。「パッケージ版が正」の注記を入れるか削除の検討を。同様に **`matsym/matsym/modulation.py` は修正前の古いコピーで §1.3 の振動数バグをまだ含む** — 未使用なら削除、使うなら同期。

### コード品質
4. **モジュール重複の解消**: `crystal_orbital_spgrep.py` と `orbital_hybridization_spgrep.py` にほぼ同一の `CrystalOrbital` クラスが 2 つある(指標照合ロジックも重複)。`vibration_modes` 側の照合(overlap 方式)も含め、ラベリング機構を 1 箇所に統合したい。
5. **`_find_intertwiner` の置き場所**: 現在 `modulation.py` にあり `phonon_vector.py` が import。群論ヘルパーとして `operations.py` へ移す方が自然。
6. **testsuite の実行時間**: 310 チェックで数分。重い計算のキャッシュか高速サブセット(smoke test)の定義でイテレーションが速くなる。

### 機能拡張(2026-07-08 ミーティングのロードマップ含む)
7. 電子軌道の**空間群 direct product**(小磯くんの PLAN.md 仕様が `crystod-group --product` を拡張する形で入る予定)
8. `crystod-mag`: **QE インプット生成の充実**(MAGMOM 相当の自動生成は優先機能)
9. `crystod-md`: **LAMMPS 対応**(`--format` の選択肢として計画済み)
10. `crystod-phonon`: 機械学習力場(MLFF)連携(優先度低)
11. 指標表の AI スキャン読取
12. `--spin-basis`/SALC 統合の一般化(AlNi3 で確立した反強磁性配列探索を他 Wyckoff 位置・高次多極子へ)
13. **`--modulation` の q 点ラベル入力**: `--vector`/`--vibration` は `--qpoint X` ができるが `--modulation` は数値のみ。ラベル対応で一貫する。
14. **マルチ arm 出力のファイル名衝突**: 同じ star の複数 arm を出力すると q ラベルが同じでファイル名が衝突し得る。arm 番号などの識別子を付ける案。
15. **非特殊 q 点のラベル**: ~~DT (Δ) 線上など特殊「点」にない q は `-` に落ちる。~~ → **7/22 に電子系(`crystod --element/--orbital`)は ISO-IR fallback で解決**(§3 参照。irreptables には線・面が無いため ISO-IR テーブルを採用)。**7/22 (2) に全経路へ展開済み**: `orbital_hybridization_spgrep.py`(--atomic-orbital)、`vibration_modes.py`(crystod-mag / --vibration)、`phonon_irreps.py`(--irreps/--vector/--modulation)。phonopy 経路はバンド組指標が偶然縮退で可約になり得るため、1 対 1 照合でなく分解型照合(`isoir.decompose_characters`)を使う。
16. **spinor(二重群)での arm 対応の検証**: 共役輸送は spinor 表現では位相規約(±E)が絡む。`--spinor` 系で arm を使うケースが出たら要検証。

### 可視化
17. HTML のさらなる軽量化: 球面調和関数を JS 側で生成し、座標でなく係数のみ埋め込む。
18. ビューワーの発展(phononwebsite 路線を維持): three.js 化、アニメーション、GIF 出力。

### リリース
19. 論文投稿(JOSS または計算材料系ジャーナル)→ オープンライセンスで pip 公開。パッケージコンセプトは「新規理論なしの、対称性解析の負担を減らすキュレーション型ツールキット」。

---

## 付録: 重要ファイルマップ

| パス | 内容 |
|---|---|
| `crystod/cli/` | セクションコマンド群(main.py = SALC 本体、bz/md/mag/phonon/group.py、common.py = 共有オプション) |
| `crystod/operations.py` | 群論ヘルパー(Wigner-D、`find_star_arm`、`conjugated_little_group_map`、`snap_qpoint` 等) |
| `crystod/isoir.py` | ISO-IR (ISOTROPY) テーブルのパーサ+非特殊 k 点の Miller–Love ラベラー(`IsoIRLabeler`) |
| `ISOTROPY/` | Stokes–Campbell ISO-IR 生データ(CIR/PIR、2011 版。iso.byu.edu/irtables.php 由来) |
| `crystod/phonon_vector.py` | 対称性適応フォノン固有ベクトル(`build_symmetry_adapted_modes` — modulation もこれを使う) |
| `crystod/phonon_irreps.py` | フォノン irrep ラベリング |
| `crystod/modulation.py` | 変調構造生成(v0.3.1 で固有ベクトル構成を phonon_vector と統一)、`_find_intertwiner` |
| `crystod/basis_function.py` | 多項式基底の irrep 分類(点群・空間群、軸性ベクトル Rx/Ry/Rz 対応) |
| `crystod/spin_basis.py` | SAMM スピン多極子基底、`_realify`、MAGMOM/QE 出力 |
| `crystod/visualize_basis.py` | SALC HTML ビューワー(plotly、Miranda 風) |
| `testsuite.py` | 回帰テスト(27 セクション・310 チェック) |
| `PLAN.md` | v0.2.0 リファクタリング計画(phonopy 依存削減、`_core/` 構想) |
| `script/` | 移植元の個別 script(小磯くん・佐藤さん作。パッケージ側が正) |
| `reference/vesta_element_rgb.json` | VESTA 元素色定義 |

---

*記録者: Claude Fable 5(2026-07-11)。v0.3.1 セッションの記録に v0.2.x〜v0.3.0 と 7/11 の f 軌道資料セッションを統合した全期間版。*
