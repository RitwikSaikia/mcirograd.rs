set shell := ["bash", "-c"]

base_dir := justfile_directory()
target_dir := base_dir / "target"

bash_args := "-euo pipefail"

default:
  #!/usr/bin/env bash
  set {{bash_args}}

  just --list

check: format build test clippy

build: format
  #!/usr/bin/env bash
  set {{bash_args}}

  cargo build

test:
  #!/usr/bin/env bash
  set {{bash_args}}

  cargo test --workspace --all-features

format:
  #!/usr/bin/env bash
  set {{bash_args}}

  cargo fmt --all

clippy:
  #!/usr/bin/env bash
  set {{bash_args}}

  cargo clippy --workspace --all-features \
    -- -D warnings

clean:
  #!/usr/bin/env bash
  set {{bash_args}}

  function clean_path {
    path=$1
    rm -rf "{{base_dir}}/$path"
  }

  if [ -d "{{base_dir}}/target" ]; then
    find "{{base_dir}}/target" -name 'libmicrograd*' \
      | sort -u \
      | xargs -I '{}' rm -rf {}
    find "{{base_dir}}/target" -name 'micrograd*' \
      | sort -u \
      | xargs -I '{}' rm -rf {}
  fi

  clean_path target/artifacts
