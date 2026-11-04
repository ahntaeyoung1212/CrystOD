# LICENSES クロスチェック（クラウド側・独立調査）

- 作成日: 2026-11-04
- 対象: ローカル側 `docs/LICENSES.md`（dist-info 由来）と突き合わせるための独立表。固定バージョンごとに PyPI の JSON メタデータ、PyPI 配布物（wheel/sdist）に同梱されたライセンスファイル、各プロジェクトの GitHub リポジトリにおける該当バージョンのタグの LICENSE を照合した。
- 注意: 本書は法的助言ではない。SPDX 表記は確認できた条文に基づく整理であり、推測で埋めた箇所は無い。確認できなかった項目は【未確認】と明記した。

## 0. 調査方法と到達性

| 情報源 | 到達 | 使い方 |
|---|---|---|
| `https://pypi.org/pypi/<pkg>/<ver>/json` | 可 | 94 件すべて HTTP 200。`license_expression`、`license`、`license_files`、分類子、`project_urls` を取得 |
| `https://files.pythonhosted.org/` | 可 | 94 件の wheel（macOS arm64 用 cp312/abi3 を優先、無ければ py3-none-any）と sentencepiece の sdist を取得し、ライセンス系ファイルと同梱バイナリ（dylib/so/metallib）を列挙。tokenizers / safetensors / hf-xet / cryptography / pillow の wheel は CycloneDX SBOM を同梱しており、依存 crate や同梱ライブラリの一覧に用いた |
| `https://raw.githubusercontent.com/` | 可 | 各プロジェクトの該当バージョンのタグ（`v<ver>`、`<ver>`、`apache-arrow-<ver>`、`python-v<ver>`、`release_v<ver>` 等）で LICENSE を取得。タグが特定できない場合は根拠欄に wheel を示した |
| `huggingface.co`、`hf-mirror.com`、`modelscope.cn`、`api.github.com`、`github.com`（HTML） | 不可（名前解決不可または 403） | モデルのライセンス条文は直接取得できず。ウェブ検索結果（検索エンジンの要約と抜粋）のみ参照し、二次情報として明記した |

SPDX の決め方: PyPI の `license_expression` があればそれを採用し、無い場合は wheel 同梱の LICENSE 本文と GitHub タグの LICENSE 本文から判定した。三者が食い違う場合は注意欄に記載した。

## 1. アプリ用 venv（Python 3.12.13、86 パッケージ）

| パッケージ | バージョン | ライセンス（SPDX） | 根拠 URL | 注意 |
|---|---|---|---|---|
| aiohappyeyeballs | 2.7.1 | PSF-2.0 | [PyPI](https://pypi.org/project/aiohappyeyeballs/2.7.1/) · [LICENSE@v2.7.1](https://raw.githubusercontent.com/aio-libs/aiohappyeyeballs/v2.7.1/LICENSE) |  |
| aiohttp | 3.14.5 | Apache-2.0 AND MIT | [PyPI](https://pypi.org/project/aiohttp/3.14.5/) · [LICENSE.txt@v3.14.5](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.14.5/LICENSE.txt) | 同梱 llhttp は MIT（wheel の licenses/vendor/llhttp/LICENSE）。 |
| aiosignal | 1.4.0 | Apache-2.0 | [PyPI](https://pypi.org/project/aiosignal/1.4.0/) · [LICENSE@v1.4.0](https://raw.githubusercontent.com/aio-libs/aiosignal/v1.4.0/LICENSE) |  |
| annotated-doc | 0.0.5 | MIT | [PyPI](https://pypi.org/project/annotated-doc/0.0.5/) · [LICENSE@0.0.5](https://raw.githubusercontent.com/fastapi/annotated-doc/0.0.5/LICENSE) |  |
| annotated-types | 0.8.0 | MIT | [PyPI](https://pypi.org/project/annotated-types/0.8.0/) · [LICENSE@v0.8.0](https://raw.githubusercontent.com/annotated-types/annotated-types/v0.8.0/LICENSE) |  |
| anyio | 4.16.0 | MIT | [PyPI](https://pypi.org/project/anyio/4.16.0/) · [LICENSE@4.16.0](https://raw.githubusercontent.com/agronholm/anyio/4.16.0/LICENSE) |  |
| attrs | 26.1.0 | MIT | [PyPI](https://pypi.org/project/attrs/26.1.0/) · [LICENSE@26.1.0](https://raw.githubusercontent.com/python-attrs/attrs/26.1.0/LICENSE) |  |
| certifi | 2026.10.23 | MPL-2.0 | [PyPI](https://pypi.org/project/certifi/2026.10.23/) · [LICENSE@2026.10.23](https://raw.githubusercontent.com/certifi/python-certifi/2026.10.23/LICENSE) | Mozilla の certdata.txt 由来の CA 束。LICENSE は MPL-2.0 表示ブロックのみ。ファイル単位コピーレフト（無改変利用に義務なし）。 |
| cffi | 2.1.1 | MIT-0 | [PyPI](https://pypi.org/project/cffi/2.1.1/) · [LICENSE@v2.1.1](https://raw.githubusercontent.com/python-cffi/cffi/v2.1.1/LICENSE) | 2.x から MIT No Attribution（表示義務なし）。wheel と GitHub v2.1.1 の LICENSE で確認。 |
| charset-normalizer | 3.5.2 | MIT | [PyPI](https://pypi.org/project/charset-normalizer/3.5.2/) · [LICENSE@3.5.2](https://raw.githubusercontent.com/jawah/charset_normalizer/3.5.2/LICENSE) |  |
| click | 8.5.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/click/8.5.0/) · [LICENSE.txt@8.5.0](https://raw.githubusercontent.com/pallets/click/8.5.0/LICENSE.txt) |  |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | [PyPI](https://pypi.org/project/cryptography/50.0.2/) · [LICENSE@50.0.2](https://raw.githubusercontent.com/pyca/cryptography/50.0.2/LICENSE) · [LICENSE.APACHE@50.0.2](https://raw.githubusercontent.com/pyca/cryptography/50.0.2/LICENSE.APACHE) · [LICENSE.BSD@50.0.2](https://raw.githubusercontent.com/pyca/cryptography/50.0.2/LICENSE.BSD) · [OpenSSL LICENSE.txt@openssl-4.0.3](https://raw.githubusercontent.com/openssl/openssl/openssl-4.0.3/LICENSE.txt) | wheel は OpenSSL 4.0.3 を静的リンク（`_rust.abi3.so` 内の版文字列と `sboms/sbom.json` で確認）。OpenSSL 4.0.3 は Apache-2.0 だが条文は wheel 非同梱。Rust 依存 39 crate は SBOM 記載、うち self_cell は Apache-2.0 OR GPL-2.0-only の選択制。 |
| datasets | 5.1.0 | Apache-2.0 | [PyPI](https://pypi.org/project/datasets/5.1.0/) · [LICENSE@5.1.0](https://raw.githubusercontent.com/huggingface/datasets/5.1.0/LICENSE) | AUTHORS 同梱。 |
| dill | 0.4.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/dill/0.4.1/) · [LICENSE@0.4.1](https://raw.githubusercontent.com/uqfoundation/dill/0.4.1/LICENSE) |  |
| fastapi | 0.162.0 | MIT | [PyPI](https://pypi.org/project/fastapi/0.162.0/) · [LICENSE@0.162.0](https://raw.githubusercontent.com/fastapi/fastapi/0.162.0/LICENSE) |  |
| filelock | 4.2.7 | MIT | [PyPI](https://pypi.org/project/filelock/4.2.7/) · [LICENSE@4.2.7](https://raw.githubusercontent.com/tox-dev/filelock/4.2.7/LICENSE) |  |
| frozenlist | 1.8.0 | Apache-2.0 | [PyPI](https://pypi.org/project/frozenlist/1.8.0/) · [LICENSE@v1.8.0](https://raw.githubusercontent.com/aio-libs/frozenlist/v1.8.0/LICENSE) |  |
| fsspec | 2026.7.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/fsspec/2026.7.0/) · [LICENSE@2026.7.0](https://raw.githubusercontent.com/fsspec/filesystem_spec/2026.7.0/LICENSE) | PyPI メタデータにライセンス欄が無い（wheel と GitHub 2026.7.0 の LICENSE で確認）。 |
| h11 | 0.16.0 | MIT | [PyPI](https://pypi.org/project/h11/0.16.0/) · [LICENSE.txt@v0.16.0](https://raw.githubusercontent.com/python-hyper/h11/v0.16.0/LICENSE.txt) |  |
| hf-transfer | 0.1.9 | Apache-2.0 | [PyPI](https://pypi.org/project/hf-transfer/0.1.9/) · [LICENSE@v0.1.9](https://raw.githubusercontent.com/huggingface/hf_transfer/v0.1.9/LICENSE) | PyPI メタデータにライセンス欄が無い。Rust 依存の一覧（SBOM）は wheel に無し。 |
| hf-xet | 1.8.0 | Apache-2.0 | [PyPI](https://pypi.org/project/hf-xet/1.8.0/) · [LICENSE@v1.8.0](https://raw.githubusercontent.com/huggingface/xet-core/v1.8.0/LICENSE) | SBOM に Rust 243 crate。MPL-2.0（colored、option-ext）、ISC/Apache-2.0（ring、aws-lc-rs/aws-lc-sys、rustls）、Unicode-3.0（ICU 系）、Zlib を含む。各 crate の条文は wheel 非同梱。→§5.5 |
| httpcore | 1.0.9 | BSD-3-Clause | [PyPI](https://pypi.org/project/httpcore/1.0.9/) · [LICENSE.md@1.0.9](https://raw.githubusercontent.com/encode/httpcore/1.0.9/LICENSE.md) |  |
| httpcore2 | 2.16.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/httpcore2/2.16.0/) · [LICENSE.md@v2.16.0](https://raw.githubusercontent.com/pydantic/httpx2/v2.16.0/LICENSE.md) | Pydantic Services による httpcore のフォーク。著作権表示は Pydantic Services と Encode の併記。 |
| httpx | 0.28.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/httpx/0.28.1/) · [LICENSE.md@0.28.1](https://raw.githubusercontent.com/encode/httpx/0.28.1/LICENSE.md) |  |
| httpx2 | 2.16.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/httpx2/2.16.0/) · [LICENSE.md@v2.16.0](https://raw.githubusercontent.com/pydantic/httpx2/v2.16.0/LICENSE.md) | Pydantic Services による httpx のフォーク（同上）。 |
| huggingface-hub | 2.4.0 | Apache-2.0 | [PyPI](https://pypi.org/project/huggingface-hub/2.4.0/) · [LICENSE@v2.4.0](https://raw.githubusercontent.com/huggingface/huggingface_hub/v2.4.0/LICENSE) |  |
| idna | 3.20 | BSD-3-Clause | [PyPI](https://pypi.org/project/idna/3.20/) · [LICENSE.md@v3.20](https://raw.githubusercontent.com/kjd/idna/v3.20/LICENSE.md) |  |
| iniconfig | 2.3.1 | MIT | [PyPI](https://pypi.org/project/iniconfig/2.3.1/) · [LICENSE@v2.3.1](https://raw.githubusercontent.com/pytest-dev/iniconfig/v2.3.1/LICENSE) |  |
| jinja2 | 3.1.6 | BSD-3-Clause | [PyPI](https://pypi.org/project/jinja2/3.1.6/) · [LICENSE.txt@3.1.6](https://raw.githubusercontent.com/pallets/jinja/3.1.6/LICENSE.txt) | PyPI メタデータは分類子のみ。 |
| jiter | 0.18.0 | MIT | [PyPI](https://pypi.org/project/jiter/0.18.0/) · [LICENSE@v0.18.0](https://raw.githubusercontent.com/pydantic/jiter/v0.18.0/LICENSE) | Rust 拡張。依存 crate の SBOM は wheel に無し【未確認】。 |
| jsonschema | 4.26.0 | MIT | [PyPI](https://pypi.org/project/jsonschema/4.26.0/) · [COPYING@v4.26.0](https://raw.githubusercontent.com/python-jsonschema/jsonschema/v4.26.0/COPYING) | ライセンスファイル名は COPYING。 |
| jsonschema-specifications | 2025.9.1 | MIT | [PyPI](https://pypi.org/project/jsonschema-specifications/2025.9.1/) · [COPYING@v2025.9.1](https://raw.githubusercontent.com/python-jsonschema/jsonschema-specifications/v2025.9.1/COPYING) | 同梱メタスキーマ（json-schema.org）の個別表記は wheel に無し【未確認】。 |
| llguidance | 1.9.1 | MIT | [PyPI](https://pypi.org/project/llguidance/1.9.1/) · [LICENSE@v1.9.1](https://raw.githubusercontent.com/microsoft/llguidance/v1.9.1/LICENSE) | Microsoft。Rust 拡張、依存 crate の SBOM は wheel に無し【未確認】。 |
| markdown-it-py | 4.2.0 | MIT | [PyPI](https://pypi.org/project/markdown-it-py/4.2.0/) · [LICENSE@v4.2.0](https://raw.githubusercontent.com/executablebooks/markdown-it-py/v4.2.0/LICENSE) | 原作 markdown-it（JS）の MIT 表記 LICENSE.markdown-it を同梱。 |
| markupsafe | 3.0.4 | BSD-3-Clause | [PyPI](https://pypi.org/project/markupsafe/3.0.4/) · [LICENSE.txt@3.0.4](https://raw.githubusercontent.com/pallets/markupsafe/3.0.4/LICENSE.txt) |  |
| mdurl | 0.1.2 | MIT | [PyPI](https://pypi.org/project/mdurl/0.1.2/) · [LICENSE@0.1.2](https://raw.githubusercontent.com/executablebooks/mdurl/0.1.2/LICENSE) |  |
| miniaudio | 1.71 | MIT | [PyPI](https://pypi.org/project/miniaudio/1.71/) · [LICENSE@v1.71](https://raw.githubusercontent.com/irmen/pyminiaudio/v1.71/LICENSE) | LICENSE は Python バインディング（Irmen de Jong）、miniaudio（David Reid）、stb_vorbis（Sean Barrett）を一括 MIT 表記。上流 miniaudio.h 自体の二重ライセンス表記は本調査では未確認。 |
| mlx | 0.32.4 | MIT | [PyPI](https://pypi.org/project/mlx/0.32.4/) · [LICENSE@v0.32.4](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/LICENSE) | Apple。 |
| mlx-audio | 0.6.4 | MIT | [PyPI](https://pypi.org/project/mlx-audio/0.6.4/) · [LICENSE@v0.6.4](https://raw.githubusercontent.com/Blaizzy/mlx-audio/v0.6.4/LICENSE) | 同梱の各モデル README が指す重みは別ライセンス（本調査対象外）。 |
| mlx-metal | 0.32.4 | MIT | [PyPI](https://pypi.org/project/mlx-metal/0.32.4/) · [LICENSE@v0.32.4](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/LICENSE) · [ACKNOWLEDGMENTS.md@v0.32.4](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/ACKNOWLEDGMENTS.md) | wheel に libmlx.dylib、mlx.metallib、libjaccl.dylib、metal-cpp ヘッダ（Apache-2.0、LICENSE.txt 同梱）。ACKNOWLEDGMENTS は PocketFFT（BSD-3）と metal-cpp を列挙。libjaccl.dylib のライセンス表記は wheel にもリポジトリにも見当たらない【未確認】。 |
| mlx-vlm | 0.7.11 | MIT | [PyPI](https://pypi.org/project/mlx-vlm/0.7.11/) · [LICENSE@v0.7.11](https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.11/LICENSE) |  |
| multidict | 7.2.1 | Apache-2.0 | [PyPI](https://pypi.org/project/multidict/7.2.1/) · [LICENSE@v7.2.1](https://raw.githubusercontent.com/aio-libs/multidict/v7.2.1/LICENSE) |  |
| multiprocess | 0.70.19 | BSD-3-Clause | [PyPI](https://pypi.org/project/multiprocess/0.70.19/) · [COPYING@0.70.19](https://raw.githubusercontent.com/uqfoundation/multiprocess/0.70.19/COPYING) · [LICENSE@0.70.19](https://raw.githubusercontent.com/uqfoundation/multiprocess/0.70.19/LICENSE) | フォーク元 multiprocessing（R Oudkerk）の BSD-3 を COPYING として同梱。 |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [PyPI](https://pypi.org/project/numpy/2.5.4/) · [LICENSE.txt@v2.5.4](https://raw.githubusercontent.com/numpy/numpy/v2.5.4/LICENSE.txt) · [wheel LICENSE.txt（同梱物の節を含む）](https://pypi.org/project/numpy/2.5.4/#files) | PyPI license_expression。macOS wheel は libscipy_openblas64_（BSD-3）、LAPACK（BSD-3-Clause-Open-MPI）、libgfortran/libgcc_s（GPL-3.0-or-later WITH GCC-exception-3.1）、libquadmath（LGPL-2.1-or-later）を同梱し、wheel の LICENSE.txt に明記。→§5.2 |
| openai | 3.44.0 | Apache-2.0 | [PyPI](https://pypi.org/project/openai/3.44.0/) · [LICENSE@v3.44.0](https://raw.githubusercontent.com/openai/openai-python/v3.44.0/LICENSE) | vendored httpx_aiohttp（BSD-3）を同梱。 |
| opencv-python | 5.0.0.93 | MIT（ラッパー）+ Apache-2.0（OpenCV）+ 同梱バイナリ多数（LGPL-2.1/LGPL-3/GPL-2.0 を含む） | [PyPI](https://pypi.org/project/opencv-python/5.0.0.93/) · [LICENSE.txt@93](https://raw.githubusercontent.com/opencv/opencv-python/93/LICENSE.txt) · [LICENSE-3RD-PARTY.txt@93](https://raw.githubusercontent.com/opencv/opencv-python/93/LICENSE-3RD-PARTY.txt) · [README@93](https://raw.githubusercontent.com/opencv/opencv-python/93/README.md) | PyPI の license 欄は "Apache 2.0" だが wheel の LICENSE.txt は MIT（README: スクリプトは MIT、OpenCV 本体は Apache-2.0）。macOS wheel に 99 dylib。FFmpeg は LGPL-2.1、libx264/libx265/librubberband は GPL-2.0-or-later で、後者は LICENSE-3RD-PARTY.txt に記載が無い。→§5.1 |
| opentelemetry-api | 1.45.2 | Apache-2.0 | [PyPI](https://pypi.org/project/opentelemetry-api/1.45.2/) · [LICENSE@v1.45.2](https://raw.githubusercontent.com/open-telemetry/opentelemetry-python/v1.45.2/LICENSE) |  |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | [PyPI](https://pypi.org/project/packaging/26.3/) · [LICENSE@26.3](https://raw.githubusercontent.com/pypa/packaging/26.3/LICENSE) · [LICENSE.APACHE@26.3](https://raw.githubusercontent.com/pypa/packaging/26.3/LICENSE.APACHE) · [LICENSE.BSD@26.3](https://raw.githubusercontent.com/pypa/packaging/26.3/LICENSE.BSD) |  |
| pandas | 3.1.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/pandas/3.1.0/) · [LICENSE@v3.1.0](https://raw.githubusercontent.com/pandas-dev/pandas/v3.1.0/LICENSE) | LICENSES/ に同梱物の条文（ultrajson BSD-3、dateutil、PSF、numpy、musl MIT、bottleneck BSD-2、pyperclip BSD-3、haven/statsbase/pyupgrade/klib/sas7bdat MIT、xsimd BSD-3、packaging Apache-2.0、fast_float Apache-2.0/BSL/MIT）。 |
| pillow | 13.0.0 | MIT-CMU | [PyPI](https://pypi.org/project/pillow/13.0.0/) · [LICENSE@13.0.0](https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/LICENSE) | wheel の SBOM（24 件）に libimagequant 4.4.1 = GPL-3.0-or-later（`_imaging` に HAVE_LIBIMAGEQUANT、静的リンク）と FriBiDi = LGPL-2.1-or-later（dylib は非同梱、実行時 dlopen）。GPL-3 条文は wheel の LICENSE に無い。FreeType（FTL）、HarfBuzz（MIT）、libjpeg-turbo（IJG/BSD-3）、libtiff、libwebp、OpenJPEG、lcms2、zlib-ng、libavif、libxcb を同梱。→§5.3 |
| pluggy | 1.7.0 | MIT | [PyPI](https://pypi.org/project/pluggy/1.7.0/) · [LICENSE@1.7.0](https://raw.githubusercontent.com/pytest-dev/pluggy/1.7.0/LICENSE) |  |
| propcache | 0.5.4 | Apache-2.0 | [PyPI](https://pypi.org/project/propcache/0.5.4/) · [LICENSE@v0.5.4](https://raw.githubusercontent.com/aio-libs/propcache/v0.5.4/LICENSE) · [NOTICE@v0.5.4](https://raw.githubusercontent.com/aio-libs/propcache/v0.5.4/NOTICE) | NOTICE 同梱。 |
| pyarrow | 26.0.1 | Apache-2.0 | [PyPI](https://pypi.org/project/pyarrow/26.0.1/) · [LICENSE.txt@apache-arrow-26.0.1](https://raw.githubusercontent.com/apache/arrow/apache-arrow-26.0.1/LICENSE.txt) · [NOTICE.txt@apache-arrow-26.0.1](https://raw.githubusercontent.com/apache/arrow/apache-arrow-26.0.1/NOTICE.txt) | NOTICE.txt（"This product includes…" 多数）と 2,265 行の LICENSE.txt を同梱。macOS wheel の dylib は libarrow*/libparquet のみで依存は静的リンク。LICENSE.txt に OpenSSL、snappy、zstd、lz4、brotli、thrift、re2、gRPC、protobuf、zlib、jemalloc、boost、ORC、flatbuffers、RapidJSON の記載あり。AWS/Azure/GCS SDK、abseil、utf8proc の記載は無い（同梱の有無も未検証）【未確認】。 |
| pycparser | 3.12 | BSD-3-Clause | [PyPI](https://pypi.org/project/pycparser/3.12/) · [LICENSE@release_v3.12](https://raw.githubusercontent.com/eliben/pycparser/release_v3.12/LICENSE) |  |
| pydantic | 2.14.2 | MIT | [PyPI](https://pypi.org/project/pydantic/2.14.2/) · [LICENSE@v2.14.2](https://raw.githubusercontent.com/pydantic/pydantic/v2.14.2/LICENSE) |  |
| pydantic-core | 2.50.2 | MIT | [PyPI](https://pypi.org/project/pydantic-core/2.50.2/) · [wheel: pydantic_core-2.50.2.dist-info/licenses/LICENSE](https://pypi.org/project/pydantic-core/2.50.2/#files) | PyPI の project_urls が pydantic/pydantic を指し、v2.50.2 タグの LICENSE は GitHub で特定できず（wheel の LICENSE は MIT、Samuel Colvin 2022）。Rust 依存の SBOM 無し【未確認】。 |
| pygments | 2.21.0 | BSD-2-Clause | [PyPI](https://pypi.org/project/pygments/2.21.0/) · [LICENSE@2.21.0](https://raw.githubusercontent.com/pygments/pygments/2.21.0/LICENSE) |  |
| pytest | 9.2.0 | MIT | [PyPI](https://pypi.org/project/pytest/9.2.0/) · [LICENSE@9.2.0](https://raw.githubusercontent.com/pytest-dev/pytest/9.2.0/LICENSE) |  |
| python-dateutil | 4.0.0 | BSD-3-Clause（2017-12 以降の寄稿は Apache-2.0 も併記） | [PyPI](https://pypi.org/project/python-dateutil/4.0.0/) · [LICENSE@4.0.0](https://raw.githubusercontent.com/dateutil/dateutil/4.0.0/LICENSE) | LICENSE は Apache-2.0 条項の後に BSD-3 全文があり「The above BSD License Applies to all code」と明記。PyPI license_expression は BSD-3-Clause。 |
| python-multipart | 0.0.33 | Apache-2.0 | [PyPI](https://pypi.org/project/python-multipart/0.0.33/) · [LICENSE.txt@0.0.33](https://raw.githubusercontent.com/Kludex/python-multipart/0.0.33/LICENSE.txt) |  |
| pyyaml | 6.0.3 | MIT | [PyPI](https://pypi.org/project/pyyaml/6.0.3/) · [LICENSE@6.0.3](https://raw.githubusercontent.com/yaml/pyyaml/6.0.3/LICENSE) |  |
| referencing | 0.37.0 | MIT | [PyPI](https://pypi.org/project/referencing/0.37.0/) · [COPYING@v0.37.0](https://raw.githubusercontent.com/python-jsonschema/referencing/v0.37.0/COPYING) |  |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | [PyPI](https://pypi.org/project/regex/2026.9.29/) · [LICENSE.txt@2026.9.29](https://raw.githubusercontent.com/mrabarnett/mrab-regex/2026.9.29/LICENSE.txt) | CPython 2.6/3.1 の re モジュール由来部分は CNRI Python 1.6 ライセンス、追加分は Apache-2.0（LICENSE.txt 冒頭）。 |
| requests | 2.34.2 | Apache-2.0 | [PyPI](https://pypi.org/project/requests/2.34.2/) · [LICENSE@v2.34.2](https://raw.githubusercontent.com/psf/requests/v2.34.2/LICENSE) · [NOTICE@v2.34.2](https://raw.githubusercontent.com/psf/requests/v2.34.2/NOTICE) | NOTICE（"Requests Copyright 2019 Kenneth Reitz"）同梱。 |
| rich | 15.0.0 | MIT | [PyPI](https://pypi.org/project/rich/15.0.0/) · [LICENSE@v15.0.0](https://raw.githubusercontent.com/Textualize/rich/v15.0.0/LICENSE) |  |
| rpds-py | 2026.9.1 | MIT | [PyPI](https://pypi.org/project/rpds-py/2026.9.1/) · [LICENSE@v2026.9.1](https://raw.githubusercontent.com/crate-py/rpds/v2026.9.1/LICENSE) | GitHub は crate-py/rpds（v2026.9.1）。Rust 依存の SBOM 無し【未確認】。 |
| safetensors | 0.8.0 | Apache-2.0 | [PyPI](https://pypi.org/project/safetensors/0.8.0/) · [LICENSE@v0.8.0](https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/LICENSE) | PyPI メタデータは分類子のみ。SBOM に Rust 40 crate（MIT/Apache-2.0/Zlib/Unicode-3.0）。 |
| scipy | 1.18.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/scipy/1.18.1/) · [LICENSE.txt@v1.18.1](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/LICENSE.txt) · [LICENSES_bundled.txt@v1.18.1](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/LICENSES_bundled.txt) | macOS wheel は libscipy_openblas（BSD-3）、LAPACK（BSD-3-Clause-Open-MPI）、libgfortran/libgcc_s（GPL-3.0-or-later WITH GCC-exception-3.1）、libquadmath（LGPL-2.1-or-later）を同梱し LICENSE.txt に明記。LICENSES_bundled に Qhull、Boost BSL-1.0、HiGHS MIT、SuperLU/ARPACK BSD-3 等。→§5.2 |
| sentencepiece | 0.2.2 | Apache-2.0 | [PyPI](https://pypi.org/project/sentencepiece/0.2.2/) · [LICENSE@v0.2.2](https://raw.githubusercontent.com/google/sentencepiece/v0.2.2/LICENSE) · [sdist third_party/*/LICENSE](https://pypi.org/project/sentencepiece/0.2.2/#files) | wheel にライセンスファイルが一切無い（PyPI license_expression と GitHub v0.2.2 LICENSE で確認）。sdist の third_party: abseil-cpp（Apache-2.0）、darts_clone（BSD-2）、esaxx（MIT）、protobuf-lite（BSD-3）が拡張に静的リンク。→§5.6 |
| shellingham | 1.5.4 | ISC | [PyPI](https://pypi.org/project/shellingham/1.5.4/) · [LICENSE@1.5.4](https://raw.githubusercontent.com/sarugaku/shellingham/1.5.4/LICENSE) |  |
| sniffio | 1.3.1 | MIT OR Apache-2.0 | [PyPI](https://pypi.org/project/sniffio/1.3.1/) · [LICENSE@v1.3.1](https://raw.githubusercontent.com/python-trio/sniffio/v1.3.1/LICENSE) · [LICENSE.MIT@v1.3.1](https://raw.githubusercontent.com/python-trio/sniffio/v1.3.1/LICENSE.MIT) |  |
| sounddevice | 0.5.6 | MIT | [PyPI](https://pypi.org/project/sounddevice/0.5.6/) · [LICENSE@0.5.6](https://raw.githubusercontent.com/spatialaudio/python-sounddevice/0.5.6/LICENSE) · [wheel(macOS universal2): _sounddevice_data/portaudio-binaries/README.md](https://pypi.org/project/sounddevice/0.5.6/#files) | macOS 用 wheel（universal2）は PortAudio の libportaudio.dylib を同梱。同梱 README.md は「PortAudio by Ross Bencina and Phil Burk, MIT License」と表記するのみで PortAudio の条文は無い。Windows 用 DLL（ASIO 版は Steinberg ASIO SDK を含む）も同じ wheel に入る。→§5.7 |
| starlette | 1.10.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/starlette/1.10.0/) · [LICENSE.md@1.10.0](https://raw.githubusercontent.com/Kludex/starlette/1.10.0/LICENSE.md) |  |
| tokenizers | 0.23.3 | Apache-2.0 | [PyPI](https://pypi.org/project/tokenizers/0.23.3/) · [LICENSE@v0.23.3](https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.3/LICENSE) | wheel にライセンスファイルが無い（SBOM のみ。Rust 127 crate、MIT/Apache-2.0 中心、unicode-ident は Unicode-3.0）。PyPI は分類子のみ、GitHub v0.23.3 LICENSE で確認。→§5.6 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | [PyPI](https://pypi.org/project/tqdm/4.70.1/) · [LICENCE@v4.70.1](https://raw.githubusercontent.com/tqdm/tqdm/v4.70.1/LICENCE) | LICENCE: 全体 MIT、Casper da Costa-Luis の寄稿分（files: *）は MPL-2.0。 |
| transformers | 5.22.1 | Apache-2.0 | [PyPI](https://pypi.org/project/transformers/5.22.1/) · [LICENSE@v5.22.1](https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/LICENSE) |  |
| truststore | 0.10.5 | MIT | [PyPI](https://pypi.org/project/truststore/0.10.5/) · [LICENSE@v0.10.5](https://raw.githubusercontent.com/sethmlarson/truststore/v0.10.5/LICENSE) |  |
| typer | 0.27.3 | MIT | [PyPI](https://pypi.org/project/typer/0.27.3/) · [LICENSE@0.27.3](https://raw.githubusercontent.com/fastapi/typer/0.27.3/LICENSE) | vendored click（BSD-3）を typer/_click/LICENSE.txt として同梱。 |
| typing-extensions | 4.16.0 | PSF-2.0 | [PyPI](https://pypi.org/project/typing-extensions/4.16.0/) · [LICENSE@4.16.0](https://raw.githubusercontent.com/python/typing_extensions/4.16.0/LICENSE) |  |
| typing-inspection | 0.4.4 | MIT | [PyPI](https://pypi.org/project/typing-inspection/0.4.4/) · [LICENSE@v0.4.4](https://raw.githubusercontent.com/pydantic/typing-inspection/v0.4.4/LICENSE) |  |
| tzdata | 2026.6 | Apache-2.0（パッケージ）/ IANA tz データはパブリックドメイン | [PyPI](https://pypi.org/project/tzdata/2026.6/) · [LICENSE@2026.6](https://raw.githubusercontent.com/python/tzdata/2026.6/LICENSE) · [IANA tz LICENSE@2026f](https://raw.githubusercontent.com/eggert/tz/2026f/LICENSE) | wheel の tzdata.zi は "# version 2026f"。IANA の LICENSE（2026f）: "all files in the tz code and data ... are in the public domain"。その条文は wheel 非同梱。 |
| urllib3 | 2.8.0 | MIT | [PyPI](https://pypi.org/project/urllib3/2.8.0/) · [LICENSE.txt@2.8.0](https://raw.githubusercontent.com/urllib3/urllib3/2.8.0/LICENSE.txt) |  |
| uvicorn | 0.54.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/uvicorn/0.54.0/) · [LICENSE.md@0.54.0](https://raw.githubusercontent.com/Kludex/uvicorn/0.54.0/LICENSE.md) |  |
| websockets | 17.4 | BSD-3-Clause | [PyPI](https://pypi.org/project/websockets/17.4/) · [LICENSE@17.4](https://raw.githubusercontent.com/python-websockets/websockets/17.4/LICENSE) |  |
| xxhash | 4.0.2 | BSD-2-Clause | [PyPI](https://pypi.org/project/xxhash/4.0.2/) · [LICENSE@v4.0.2](https://raw.githubusercontent.com/ifduyue/python-xxhash/v4.0.2/LICENSE) | 同梱 xxHash C ライブラリ（BSD-2）の個別条文は wheel LICENSE に無い【未確認】。 |
| yarl | 1.25.1 | Apache-2.0 | [PyPI](https://pypi.org/project/yarl/1.25.1/) · [LICENSE@v1.25.1](https://raw.githubusercontent.com/aio-libs/yarl/v1.25.1/LICENSE) · [NOTICE@v1.25.1](https://raw.githubusercontent.com/aio-libs/yarl/v1.25.1/NOTICE) | NOTICE 同梱。 |

## 1b. ツール実行用 venv（runtime/venv）

numpy 2.5.4、pandas 3.1.0、pillow 13.0.0、scipy 1.18.1、tzdata 2026.6、packaging 26.3、python-dateutil 4.0.0 はアプリ用 venv と同一バージョンなので §1 の行を参照。以下は runtime/venv にのみ入る 8 件。

| パッケージ | バージョン | ライセンス（SPDX） | 根拠 URL | 注意 |
|---|---|---|---|---|
| contourpy | 1.4.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/contourpy/1.4.0/) · [LICENSE@v1.4.0](https://raw.githubusercontent.com/contourpy/contourpy/v1.4.0/LICENSE) |  |
| cycler | 0.12.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/cycler/0.12.1/) · [LICENSE@v0.12.1](https://raw.githubusercontent.com/matplotlib/cycler/v0.12.1/LICENSE) | PyPI の license 欄は BSD-3 全文。 |
| et-xmlfile | 2.0.0 | MIT | [PyPI](https://pypi.org/project/et-xmlfile/2.0.0/) · [wheel: et_xmlfile-2.0.0.dist-info/LICENCE.rst](https://pypi.org/project/et-xmlfile/2.0.0/#files) | 標準ライブラリ由来コードに PSF-2.0（wheel の LICENCE.python）。リポジトリは Heptapod（GitHub 無し）。 |
| fonttools | 4.68.0 | MIT | [PyPI](https://pypi.org/project/fonttools/4.68.0/) · [LICENSE@4.68.0](https://raw.githubusercontent.com/fonttools/fonttools/4.68.0/LICENSE) · [LICENSE.external@4.68.0](https://raw.githubusercontent.com/fonttools/fonttools/4.68.0/LICENSE.external) | LICENSE.external はテスト用フォント（OFL-1.1 等）の表記。wheel にテストフォントは含まれない。 |
| kiwisolver | 1.5.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/kiwisolver/1.5.1/) · [LICENSE@1.5.1](https://raw.githubusercontent.com/nucleic/kiwi/1.5.1/LICENSE) |  |
| matplotlib | 3.11.2 | matplotlib License（PSF 系、SPDX 固有 ID 無し） | [PyPI](https://pypi.org/project/matplotlib/3.11.2/) · [LICENSE/LICENSE@v3.11.2](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.11.2/LICENSE/LICENSE) | PyPI 分類は Python Software Foundation License。wheel の dist-info/LICENSE は同梱物の条文を連結（STIX/LastResort OFL-1.1、BaKoMa Fonts Licence、FreeType FTL OR GPL-2.0-or-later、HarfBuzz MIT-Modern-Variant、SheenBidi Apache-2.0、libraqm MIT、Qhull、ColorBrewer Apache-2.0、Solarized MIT、JSXTools CC0-1.0、Yorick BSD 系）。DejaVu は fonts/ttf/LICENSE_DEJAVU。→§5.4 |
| openpyxl | 3.1.5 | MIT | [PyPI](https://pypi.org/project/openpyxl/3.1.5/) · [wheel: openpyxl-3.1.5.dist-info/LICENCE.rst](https://pypi.org/project/openpyxl/3.1.5/#files) | リポジトリは Heptapod（GitHub 無し）。 |
| pyparsing | 3.3.3 | MIT | [PyPI](https://pypi.org/project/pyparsing/3.3.3/) · [LICENSE@3.3.3](https://raw.githubusercontent.com/pyparsing/pyparsing/3.3.3/LICENSE) |  |


GUI 同梱 JS（marked、DOMPurify、highlight.js、diff2html、KaTeX）は依頼範囲外のため本書では再確認していない。

## 2. モデルのライセンス

### 2.1 取得状況

- 公式 `Qwen/Qwen3.8-Flash-Next`（revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`）の LICENSE と、`mlx-community/Qwen3.8-Flash-Next-4bit`（revision `07b5dc6c54600a359b87f1e53e7adf6351c72a2c`）の README／メタデータは、本セッションから huggingface.co に到達できないため条文を直接確認できなかった【未確認】。ローカル側で確認すべき URL:
  - `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE`
  - `https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit/tree/07b5dc6c54600a359b87f1e53e7adf6351c72a2c`（README の `license` / `license_name` と LICENSE ファイルの有無）
- GitHub の公式リポジトリ `QwenLM/Qwen3.8` は LICENSE が Apache-2.0（取得済み、`https://raw.githubusercontent.com/QwenLM/Qwen3.8/main/LICENSE`）だが、README の「License Agreement」節は「Please find the license file released with the model weights on Hugging Face Hub or ModelScope」とあり、Apache-2.0 はコード側のライセンスで重みには適用されない。GitHub の org 一覧で同リポジトリに付く「Apache License 2.0」ラベルを重みのライセンスと誤読しないこと。

### 2.2 ウェブ検索結果から読み取れる内容（二次情報。条文照合は【未確認】）

複数の検索結果（公式 LICENSE ページの抜粋、派生モデルのカード、解説記事）で一致していた点:

- 名称は「Qwen Community License 1.0」、冒頭は「Copyright (c) 2026 Qwen」。MIT 型の許諾文（use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, create derivative works）で、対象は重み、設定ファイル、推論コード、ドキュメント。
- 条件 1（表示義務）: 月間アクティブユーザー 1 億超、または月間売上 2,000 万 US ドル超の商用製品・サービスで使う場合、UI 上にモデル名を目立つ形で表示する。
- 条件 2（別途ライセンス）: ライセンシーまたはその関連会社が「Model-as-a-Service」または「AI Work Assistant」事業を営む場合、商用利用の前に Qwen から別途ライセンスを得る（連絡先 model-business@notice.qwencloud.com）。内部利用（ソフトウェア、その出力、基盤モデルの能力を第三者に提供しない利用）は条件 2 の対象外。
- 定義（抜粋）: Model-as-a-Service は第三者に推論やファインチューニングのアクセスを与え、入力・パラメータ・学習データを実質的に制御させるもの。AI Work Assistant は AI 支援コーディングや業務生産性を主目的とする独立製品（例として Qoder、QwenWork）。
- 法令遵守と第三者の知的財産権の不侵害、および無保証・免責の条項がある。

検索結果間で一致しない、または 1 件しか根拠が無い点:

- 条件 1 の閾値（1 億 MAU／2,000 万 US ドル）は派生モデルのカードと解説記事に基づく。条件 2 には売上閾値が無いとする記事と、Qwen3.8-Max 側の別契約（12 か月累計 5,000 万 US ドル）と混同している記事がある。
- 「翻訳ツールやショッピングアシスタント等の単一目的ツール・領域特化アシスタントは AI Work Assistant から除外（(a)(b)(c)）」は 1 件の検索要約にしか現れない。
- 複製物への著作権表示・ライセンス文の同梱義務（MIT 型なら通常ある）の正確な文言は確認できなかった。
- Acceptable Use Policy や禁止用途条項の有無は確認できなかった。

### 2.3 mlx-community 側の表記

- `mlx-community/Qwen3.8-Flash-Next-4bit` の README／メタデータの `license` 値は未確認【未確認】。
- 同じ重みの他の MLX 変換（rapid-mlx、Vontra、TensorFold 等）のカードは「Qwen/Qwen3.8-Flash-Next の派生物であり Qwen Community License 1.0 で配布」と記載している（検索結果の要約）。また一部の派生リポジトリは誤って apache-2.0 タグを付けているとの指摘がある。量子化版は元の重みの派生物として元ライセンスを継承するのが通常の扱いであり、mlx-community 版もそう扱うのが安全である。

### 2.4 個人がローカルで使う場合の義務（2.2 の内容が条文と一致することを前提とした暫定整理）

| 項目 | 公式重み | mlx-community 4bit 版 |
|---|---|---|
| 使用・改変・量子化・ファインチューニング | 許諾範囲内 | 同じ（派生物として同条件） |
| 表示義務 | 閾値超の商用製品のみ。個人のローカル利用では発生しない | 同じ |
| 利用制限 | 法令遵守、第三者 IP の不侵害。MaaS／AI Work Assistant 事業者は別途ライセンス。本人だけが使う内部利用は対象外 | 同じ |
| 再配布 | 量子化版や改変版を配る場合は LICENSE（と著作権表示）を添付するのが MIT 型の通常要件（文言未確認） | 変換元の表記（Qwen Community License 1.0）を引き継ぐ |
| 月間アクティブユーザー条項 | あり（表示義務のトリガー。個人利用には無関係） | 同じ |

## 3. 個人のローカル利用で実際に必要な対応

- 配布しない限り（本人の Mac 内で動かすだけ）、MIT／BSD／Apache-2.0／MPL-2.0／LGPL／GPL のいずれも表示・NOTICE 同梱・ソース提供の義務は発生しない。これらの義務はすべて「複製物や改変物を他者に頒布するとき」の条件である。したがって現状で必須の作業は無い。
- 推奨 1: 将来の配布や共有に備え、各 venv の `*.dist-info/licenses/` と `LICENSE*`、`NOTICE*`、`sboms/*.json` を `docs/third_party/` にそのまま保存しておく（Apache-2.0 §4(d) の NOTICE 同梱、BSD／MIT の表示義務に備える）。
- 推奨 2: opencv-python を配布物に含める場合は、FFmpeg（LGPL-2.1）に加えて libx264／libx265／librubberband（GPL-2.0-or-later）が同梱されているため、配布物全体が GPL-2.0 の条件に服する可能性がある。アプリで OpenCV を実際に使わないなら venv から外すのが最も簡単な対処である（§5.1）。
- 推奨 3: GUI にはすでに「Qwen3.8-Flash-Next・ローカル実行」の表示があるため、Qwen 側の表示義務（閾値超の商用製品が対象）を仮に広く解釈しても満たしている。
- 推奨 4: モデルの LICENSE（公式 revision と mlx-community revision）をローカルで開いて 2.2 の各項目と照合し、本書の【未確認】を解消する。特に「複製物への表示義務」と「禁止用途」の有無。
- 推奨 5: Pillow の libimagequant（GPL-3.0-or-later、静的リンク）と NumPy／SciPy の libgfortran（GPL-3 with GCC exception）、libquadmath（LGPL-2.1-or-later）は、ローカル利用では義務が無い。配布する場合は wheel をそのまま再配布する形にすれば各 wheel 同梱の条文で足りるが、Pillow は GPL-3 条文を同梱していないので別途添える（§5.3）。
- sounddevice の Windows 用 ASIO DLL（Steinberg SDK）は macOS では使われない。配布物から wheel ごと除く選択肢もある（§5.7）。

## 4. 【未確認】一覧

1. 公式 `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d…` の LICENSE 条文（huggingface.co 到達不可）。
2. `mlx-community/Qwen3.8-Flash-Next-4bit` revision `07b5dc6c…` の README／メタデータのライセンス表記と LICENSE ファイルの有無。
3. Qwen Community License 1.0 の「除外 (a)(b)(c)」、複製物への表示義務の文言、禁止用途条項の有無。
4. mlx-metal 0.32.4 に同梱の `libjaccl.dylib`（JACCL）のライセンス表記。
5. opencv-python 5.0.0.93 macOS wheel に同梱されるが LICENSE-3RD-PARTY.txt に記載が無い 45 本の dylib のうち、上流条文を取得できなかった libaribb24、librist、libX11／libXdmcp。
6. pyarrow 26.0.1 の macOS wheel に AWS／Azure／GCS SDK、abseil、utf8proc が静的リンクされているか、およびそれらの表記。
7. pydantic-core 2.50.2 の GitHub タグ（PyPI の project_urls が pydantic/pydantic を指す）。
8. Rust 拡張のうち SBOM を同梱しない jiter、rpds-py、llguidance、hf-transfer、pydantic-core の依存 crate のライセンス。
9. matplotlib 3.11.2 の `mpl-data/fonts/afm/`（標準 35 書体の AFM、48 ファイル）のライセンス表記（wheel に見当たらない）。
10. xxhash 4.0.2 に同梱される xxHash C ライブラリの個別条文、jsonschema-specifications 2025.9.1 に同梱されるメタスキーマの個別条文。
11. miniaudio 1.71 の上流 miniaudio.h 自体の二重ライセンス表記（wheel の LICENSE は一括 MIT 表記）。
12. opencv-python のタグ `93` が 5.0.0.93 に対応するという対応付け（タグ `5.0.0.93` は存在せず、ビルド番号 93 のタグを採用）。

## 5. 特記事項

### 5.1 opencv-python 5.0.0.93（macOS arm64 wheel）

- PyPI の `license` 欄は "Apache 2.0"、wheel の `LICENSE.txt` は MIT（Olli-Pekka Heinisuo、ラッパースクリプト）、`LICENSE-3RD-PARTY.txt`（179,634 バイト）が OpenCV 本体（Apache-2.0）と同梱ライブラリの条文。README は「スクリプトは MIT、OpenCV 本体は Apache-2.0、全 wheel に FFmpeg（LGPLv2.1）を同梱」と説明している。
- wheel には `cv2/.dylibs/` に 99 本の dylib がある。`LICENSE-3RD-PARTY.txt` に記載があるのは 54 本（FFmpeg 系、libbluray／gnutls／nettle／hogweed／intl／mp3lame／p11／rtmp／soxr／tasn1 を LGPL-2.1 として、libgmp／idn2／unistring を LGPL-3 として記載、ほか OpenSSL、opencore-amr、fontconfig、freetype、libpng、dav1d、ffi、ogg、openjp2、opus、rav1e、snappy、speex、srt、theora、webp、vorbis、vpx、xcb 等）。
- 記載が無い 45 本: Iex／IlmThread／Imath／OpenEXR／OpenEXRCore、SDL2、SvtAv1Enc、X11、Xdmcp、aribb24、b2、brotli（3 本）、cjson、deflate、fribidi、glib、graphite2、harfbuzz、hwy、jxl（3 本）、lcms2、leptonica、lz4、lzma、mbedcrypto、p11-kit、pcre2、rist、rubberband、samplerate、sodium、ssh、tesseract、unibreak、vidstab、vmaf、x264、x265、zimg、zmq、zstd。
- このうち上流の COPYING／LICENSE を GitHub（バージョン非固定のミラーまたは本家）で確認した結果: x264 と x265 と rubberband は GPL-2.0-or-later、vid.stab は LGPL-2.1-or-later、glib・fribidi・libssh は LGPL-2.1-or-later、graphite2 は LGPL-2.1-or-later／MPL-2.0／GPL-2.0-or-later の選択制、libzmq は MPL-2.0、mbedtls は Apache-2.0 OR GPL-2.0-or-later、zimg は WTFPL、残り（tesseract Apache-2.0、leptonica 独自 BSD 系、vmaf BSD-2-Clause-Patent、SVT-AV1 BSD-3-Clause-Clear、sodium ISC、SDL2 zlib、b2 CC0-1.0、OpenEXR／jxl／p11-kit／zstd BSD-3、pcre2 BSD-3 WITH PCRE2-exception、lz4／samplerate／libarchive BSD-2、cjson／deflate／brotli／lcms2 MIT、harfbuzz Old MIT、hwy Apache-2.0 OR BSD-3、unibreak zlib 系、xz は public domain 主体、libass ISC）は許容的。
- 含意: libx264／libx265 が同梱されている以上、この FFmpeg は GPL 有効でビルドされたものと考えられ、wheel を含む配布物は GPL-2.0 の条件（ソース提供等）に服し得る。ローカル利用のみなら義務は無い。

### 5.2 numpy 2.5.4 と scipy 1.18.1（macOS arm64 wheel）

- 両 wheel とも `libgfortran.5.dylib`、`libgcc_s.1.1.dylib`、`libquadmath.0.dylib`、OpenBLAS（`libscipy_openblas64_.dylib`／`libscipy_openblas.dylib`）を同梱し、wheel の `LICENSE.txt` に「GCC runtime library: GPL-3.0-or-later WITH GCC-exception-3.1」「libquadmath: LGPL-2.1-or-later」「OpenBLAS: BSD-3-Clause」「LAPACK: BSD-3-Clause-Open-MPI」と明記している。GCC Runtime Library Exception により、これらを動的リンクした成果物に GPL は及ばない。

### 5.3 pillow 13.0.0（macOS arm64 wheel）

- `sboms/pillow-13.0.0.cdx.json` に 24 コンポーネント。libimagequant 4.4.1 は GPL-3.0-or-later と記載され、`PIL/_imaging.cpython-312-darwin.so` に `HAVE_LIBIMAGEQUANT` があるため静的リンクされている。wheel の `LICENSE`（1,574 行、同梱ライブラリの条文を連結）には GPL-3 の条文も libimagequant の記載も無い。
- FriBiDi 1.0.17（LGPL-2.1-or-later）は SBOM に載るが wheel に libfribidi の dylib は無く、`_imagingft` は実行時に `libfribidi` を dlopen する shim 方式（文字列 `libfribidi.so`、`HAVE_FRIBIDI` を確認）。
- FreeType は FTL、HarfBuzz は MIT、libjpeg-turbo は IJG と BSD-3、libtiff は libtiff ライセンス、libwebp は BSD-3、OpenJPEG と libavif は BSD-2、lcms2 と raqm は MIT、zlib-ng は Zlib、libxcb は X11、liblzma はパブリックドメイン主体。

### 5.4 matplotlib 3.11.2（フォント）

- `dist-info/LICENSE`（63,442 バイト）は matplotlib 本体のライセンスに続けて同梱物の条文を連結している。フォント関係: STIX（`fonts/ttf/STIX*.ttf`、OFL-1.1）、LastResortHE（OFL-1.1）、BaKoMa Computer Modern（`fonts/ttf/cm*.ttf` と `fonts/afm/cm*.afm`、BaKoMa Fonts Licence: 著作権表示・作者表記・許諾文を複製物に残せば複製・配布可、文書への埋め込みと印刷は表示不要）、AMS Fonts（OFL-1.1、テスト用 pfb で wheel 非同梱）、Courier 10 Pitch BT（Bitstream、同じくテスト用で非同梱）。
- DejaVu（`fonts/ttf/DejaVu*.ttf`）は dist-info の LICENSE には載らず、同じディレクトリの `LICENSE_DEJAVU`（Bitstream Vera ライセンス、DejaVu 変更分はパブリックドメイン）が根拠。
- `fonts/pdfcorefonts/*.afm`（Adobe の PDF コア 14 書体メトリクス）は同梱 `readme.txt` の Adobe 許諾文（著作権表示の保持、readme を伴う配布、改変箇所の明示）が根拠。`fonts/afm/` の 48 ファイル（標準 35 書体のメトリクス）はライセンス表記が wheel に無い【未確認】。
- 非フォント: FreeType（FTL OR GPL-2.0-or-later、`ft2font` に静的リンク）、HarfBuzz（MIT-Modern-Variant）、SheenBidi（Apache-2.0）、libraqm（MIT）、Qhull（Qhull ライセンス）、ColorBrewer（Apache-2.0）、Solarized（MIT）、JSXTools resize observer（CC0-1.0）、Yorick カラーマップ（BSD 系）。

### 5.5 Rust 拡張（tokenizers、safetensors、hf-xet、cryptography、pydantic-core、jiter、rpds-py、llguidance、hf-transfer）

- tokenizers（127 crate）、safetensors（40）、hf-xet（243）、cryptography（39）は wheel に CycloneDX SBOM を同梱し、依存 crate のライセンス式を列挙している。大半は MIT OR Apache-2.0。hf-xet には MPL-2.0（colored、option-ext）、ISC／Apache-2.0 系（ring、aws-lc-rs、aws-lc-sys、rustls、rustls-webpki）、Unicode-3.0（ICU 系 18 crate）、Zlib が含まれる。cryptography の self_cell は Apache-2.0 OR GPL-2.0-only の選択制。
- いずれの wheel も各 crate の条文そのものは同梱していない。MPL-2.0 はファイル単位のコピーレフトで、当該ファイルを改変しない限り追加義務は無い。
- SBOM を同梱しない jiter、rpds-py、llguidance、hf-transfer、pydantic-core の依存は【未確認】。

### 5.6 wheel にライセンスファイルを同梱しないもの

- sentencepiece 0.2.2: dist-info にライセンスファイルが無い。PyPI の `license_expression`（Apache-2.0）と GitHub v0.2.2 の LICENSE で確認。sdist の `third_party/`（abseil-cpp Apache-2.0、darts_clone BSD-2、esaxx MIT、protobuf-lite BSD-3）が拡張に静的に取り込まれる。
- tokenizers 0.23.3: dist-info に SBOM のみ。GitHub v0.23.3 の LICENSE（Apache-2.0）で確認。
- 配布時はこの 2 件について上流 LICENSE を別途添える必要がある。

### 5.7 sounddevice 0.5.6

- macOS 向けは `py3-none-macosx_10_6_x86_64.macosx_10_6_universal2` wheel で、`_sounddevice_data/portaudio-binaries/libportaudio.dylib` を同梱する。同梱 README.md の表記は「PortAudio by Ross Bencina and Phil Burk, MIT License」と「Steinberg Audio Stream I/O API by Steinberg Media Technologies GmbH」のみで、PortAudio の条文は含まれない。
- 同じ wheel に Windows 用 DLL（ASIO 対応版を含む）が入る。ASIO SDK は Steinberg の独自条件で、macOS では使われない。

### 5.8 その他

- mlx-metal 0.32.4: `libmlx.dylib`、`mlx.metallib`、`libjaccl.dylib`、metal-cpp ヘッダ（Apache-2.0、LICENSE.txt 同梱）。ACKNOWLEDGMENTS.md の第三者ソフトウェア節は PocketFFT（BSD-3）と metal-cpp のみ。JACCL（Apple の集団通信ライブラリ）のライセンス表記は見つからなかった【未確認】。
- cryptography 50.0.2: OpenSSL 4.0.3 を静的リンク（Apache-2.0、条文は非同梱）。
- tzdata 2026.6: パッケージは Apache-2.0、同梱データは IANA tz 2026f（`tzdata.zi` の版表記で確認）でパブリックドメイン。IANA の LICENSE は非同梱。
- certifi と tqdm と libzmq（opencv 同梱）の MPL-2.0 はファイル単位のコピーレフト。無改変利用に義務は無い。
- PyPI メタデータと実体の不一致: opencv-python（PyPI "Apache 2.0" と wheel MIT）、fsspec と hf-transfer（PyPI にライセンス欄無し）、jinja2・safetensors・tokenizers（分類子のみ）。SPDX 欄は wheel と GitHub の条文で判定した。

## 6. 主な参照 URL（モデル関連、二次情報）

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE （本セッションから到達不可）
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/discussions/18 （「Why isn't it Apache 2.0 licensed?」、到達不可）
- https://raw.githubusercontent.com/QwenLM/Qwen3.8/main/LICENSE （コード側 Apache-2.0、取得済み）
- https://raw.githubusercontent.com/QwenLM/Qwen3.8/main/README.md （「License Agreement」節、取得済み）
- https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit （到達不可）
- https://huggingface.co/rapid-mlx/Qwen3.8-Flash-Next-4bit （派生モデルのカード、検索要約のみ）
- https://cellcog.ai/blog/qwen3-8-flash-next/ 、https://inferya.com/guides/qwen-community-license-commercial-use/ （解説記事、到達不可、検索要約のみ）
